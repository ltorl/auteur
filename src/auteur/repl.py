"""The interactive session: type ideas and notes, drive the film with /commands.

Modelled on Claude Code / Codex: a persistent prompt with slash commands,
completion with descriptions, history, a status line, Ctrl-C to cancel the
running step, Ctrl-D (or /exit) to leave.
"""

from __future__ import annotations

import shlex
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import actions as A
from . import config as cfg
from . import prompts, ui
from .actions import Abort
from .llm import ProxyError, codex_auth_file, proxy_up
from .schemas import Project

# ---- command registry ----------------------------------------------------------- #


@dataclass
class Command:
    name: str
    help: str
    group: str
    fn: Callable[[Session, list[str]], None]
    usage: str = ""
    aliases: tuple[str, ...] = ()
    complete: Callable[[Session, list[str]], Iterable[str]] | None = None


COMMANDS: dict[str, Command] = {}
ALIASES: dict[str, str] = {}
GROUPS = ("Film", "Camera", "Setup", "Session")


def command(name: str, help: str, group: str, usage: str = "", aliases: tuple[str, ...] = (), complete=None):
    def deco(fn):
        COMMANDS[name] = Command(name, help, group, fn, usage, aliases, complete)
        for a in aliases:
            ALIASES[a] = name
        return fn

    return deco


def lookup(name: str) -> Command | None:
    return COMMANDS.get(ALIASES.get(name, name))


def split(line: str) -> list[str]:
    """shlex, but forgiving of a stray apostrophe ("the vault's heist")."""
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def opts(argv: list[str], values: dict[str, str] | None = None, flags: dict[str, str] | None = None):
    """Tiny flag parser: returns (positionals, options)."""
    values, flags = values or {}, flags or {}
    pos, out = [], {v: None for v in values.values()} | {v: False for v in flags.values()}
    it = iter(argv)
    for a in it:
        if a in values:
            nxt = next(it, None)
            if nxt is None:
                raise Abort(f"{a} needs a value")
            out[values[a]] = nxt
        elif a in flags:
            out[flags[a]] = True
        elif a.startswith("-") and len(a) > 1 and not a[1:].replace(".", "").isdigit():
            raise Abort(f"unknown option {a}")
        else:
            pos.append(a)
    return pos, out


# ---- the session ----------------------------------------------------------------- #


@dataclass
class Session:
    project_ref: str | None = None
    resume: bool = False
    pt_input: object | None = None  # prompt_toolkit input/output overrides (tests)
    pt_output: object | None = None
    project: Project | None = None
    running: bool = True
    _models: list[str] = field(default_factory=list)
    _online: bool = False
    _last_interrupt: float = 0.0

    def __post_init__(self):
        self.studio = A.studio()
        if self.project_ref or self.resume:
            try:
                self.project = A.find_project(self.studio, self.project_ref)
            except Abort as e:
                ui.err(str(e), e.hint)

    # -- state ------------------------------------------------------------------- #
    @property
    def settings(self) -> cfg.Settings:
        return self.studio.settings

    def reload_settings(self) -> None:
        llm = self.studio.llm
        self.studio = A.studio()
        if self.studio.settings.base_url == llm.settings.base_url:
            self.studio.llm._client = llm._client
            self.studio.llm._models = llm._models

    def need_project(self) -> Project:
        if self.project is None:
            raise Abort("no film open", "describe an idea to start one, /new, or /open <film>")
        return self.project

    def refresh(self) -> None:
        self._online = proxy_up(self.settings.base_url, timeout=0.5)
        if self.studio.llm._models:
            self._models = self.studio.llm._models

    # -- chrome ----------------------------------------------------------------- #
    def welcome(self) -> None:
        ui.banner()
        signed = codex_auth_file().exists()
        lines = Text()
        lines.append("✻ ", style="accent")
        lines.append("Welcome to Auteur", style="title")
        lines.append("  — your film crew lives in the terminal\n\n", style="dim")
        lines.append("  /help", style="accent")
        lines.append(" for commands  ·  ", style="dim")
        lines.append("/login" if not signed else "/new", style="accent")
        lines.append(" to sign in" if not signed else " for a guided start", style="dim")
        lines.append("  ·  ", style="dim")
        lines.append("Ctrl-D", style="accent")
        lines.append(" to exit\n", style="dim")
        lines.append(f"\n  films  {ui.tilde(self.settings.home_path)}", style="dim")
        ui.console.print(Panel(lines, box=box.ROUNDED, border_style=ui.AMBER, padding=(1, 2), expand=False,
                               width=min(ui.console.width, 78)))

        tips = []
        if not signed:
            tips.append(("/login", "sign in with ChatGPT — that's your director"))
        from . import weights as W

        if self.settings.backend == "airstream" and not W.present(Path(self.settings.weights).expanduser()):
            tips.append(("/fetch", "download LTX-2.5 (or /backend animatic to preview without it)"))
        tips.append(("idea", "just describe a film — \"a lighthouse keeper finds a door in the sea\""))
        recent = self.studio.store.all()[:3]
        if recent and self.project is None:
            tips.append(("/open", "pick up " + ", ".join(p.slug for p in recent)))
        ui.console.print(Text("  Tips for getting started:", style="dim"))
        for i, (cmd, tip) in enumerate(tips, 1):
            ui.console.print(Text.assemble((f"   {i}. ", "dim"), (cmd, "accent"), ("  " + tip, "ink")))
        if self.project:
            ui.console.print()
            ui.console.print(Text.assemble(("  ▶ ", "accent"), ("resuming ", "dim"), (self.project.title, "title")))
            A.next_steps(self.project)

    def toolbar(self):
        p = self.project
        if p:
            n = len(p.shots)
            shot = sum(s.status == "shot" for s in p.shots)
            film = f" ◆ {p.slug} " + (f"· {shot}/{n} shot " if n else "· treatment ")
        else:
            film = " ◆ no film "
        model = self.studio.llm.text_model if self._models else (self.settings.text_model or "auto")
        dot = "●" if self._online else "○"
        rest = f"{model} │ camera {self.settings.backend}·{self.settings.quality} │ {self.settings.aspect} "
        parts = [
            ("class:tb.film", film),
            ("class:tb", "│ director "),
            ("class:tb.on" if self._online else "class:tb.off", f"{dot} "),
            ("class:tb", rest),
        ]
        hint = "  / commands · @ shots · esc⏎ newline "
        if sum(len(t) for _, t in parts) + len(hint) <= shutil.get_terminal_size().columns:
            parts.append(("class:tb.hint", hint))
        return parts

    def placeholder(self):
        if self.project is None:
            return [("class:placeholder", "Describe a film idea, or / for commands")]
        return [("class:placeholder", f"Give the director a note on “{self.project.title}”, or / for commands")]

    # -- loop -------------------------------------------------------------------- #
    def _prompt_session(self):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style

        kb = KeyBindings()

        @kb.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        style = Style.from_dict({
            "prompt": f"{ui.AMBER} bold",
            "placeholder": ui.SMOKE,
            "bottom-toolbar": f"bg:{ui.NIGHT} {ui.SMOKE} noreverse",
            "tb": f"bg:{ui.NIGHT} {ui.SMOKE}",
            "tb.film": f"bg:{ui.AMBER} {ui.NIGHT} bold",
            "tb.on": f"bg:{ui.NIGHT} #7BD88F",
            "tb.off": f"bg:{ui.NIGHT} {ui.ROSE}",
            "tb.hint": f"bg:{ui.NIGHT} #5a564f",
            "completion-menu": f"bg:{ui.NIGHT}",
            "completion-menu.completion": f"bg:{ui.NIGHT} {ui.INK}",
            "completion-menu.completion.current": f"bg:{ui.AMBER} {ui.NIGHT} bold",
            "completion-menu.meta.completion": f"bg:{ui.NIGHT} {ui.SMOKE}",
            "completion-menu.meta.completion.current": f"bg:{ui.AMBER} {ui.NIGHT}",
        })
        try:
            path = cfg.config_path().parent / "history"
            path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(path))
        except OSError:
            history = InMemoryHistory()
        return PromptSession(
            history=history,
            completer=SlashCompleter(self),
            complete_while_typing=True,
            key_bindings=kb,
            style=style,
            bottom_toolbar=self.toolbar,
            placeholder=self.placeholder,
            input=self.pt_input,
            output=self.pt_output,
            reserve_space_for_menu=8,
        )

    def run(self) -> None:
        self.welcome()
        self.refresh()
        if self.pt_input is None and not (sys.stdin.isatty() and sys.stdout.isatty()):
            return self._run_plain()
        ps = self._prompt_session()
        while self.running:
            try:
                ui.console.print()
                line = ps.prompt([("class:prompt", "› ")])
            except KeyboardInterrupt:
                if time.time() - self._last_interrupt < 2:
                    break
                self._last_interrupt = time.time()
                ui.console.print(Text("  (press Ctrl-C again to exit)", style="dim"))
                continue
            except EOFError:
                break
            self.handle(line)
            self.refresh()
        ui.console.print(Text("\n  ✻ that's a wrap.\n", style="accent"))

    def _run_plain(self) -> None:
        """Piped input: one line per prompt, no line editor."""
        for line in sys.stdin:
            if not self.running:
                break
            ui.console.print(Text.assemble(("\n› ", "accent"), (line.rstrip(), "ink")))
            self.handle(line)
            self.refresh()

    def handle(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            if line.startswith("/"):
                name, *argv = split(line[1:]) or [""]
                cmd = lookup(name)
                if cmd is None:
                    close = [c for c in COMMANDS if c.startswith(name[:2])] if name else []
                    raise Abort(f"unknown command /{name}", ("did you mean /" + close[0] + "?") if close else "/help")
                cmd.fn(self, argv)
            elif line in ("?", "help"):
                cmd_help(self, [])
            elif line in ("exit", "quit"):
                self.running = False
            else:
                self.talk(line)
        except Abort as e:
            ui.err(str(e), e.hint)
        except ProxyError as e:
            ui.err(str(e), e.hint)
        except KeyboardInterrupt:
            ui.console.print()
            ui.warn("cut! — interrupted (progress so far is saved)")
        except Exception as e:  # noqa: BLE001 — the session outlives any one step
            ui.err(f"{type(e).__name__}: {e}")

    def talk(self, text: str) -> None:
        """Free text: a new film's idea, or a note on the open one."""
        if self.project is None:
            self.project = A.start_film(self.studio, text)
            A.next_steps(self.project)
        else:
            A.do_revise(self.studio, self.project, text)
            A.next_steps(self.project)


# ---- completion -------------------------------------------------------------------- #


class SlashCompleter(Completer):
    """/commands with descriptions, their arguments, and @shot mentions."""

    def __init__(self, session: Session):
        self.s = session

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)
        if text.startswith("/") and " " not in text:
            typed = text[1:]
            for c in COMMANDS.values():
                if c.name.startswith(typed):
                    yield Completion("/" + c.name, start_position=-len(text), display="/" + c.name,
                                     display_meta=c.help)
            return
        if text.startswith("/"):
            parts = split(text[1:])
            cmd = lookup(parts[0]) if parts else None
            if cmd and cmd.complete:
                args = parts[1:] if not text.endswith(" ") else parts[1:] + [""]
                current = "" if text.endswith(" ") else word
                for cand in cmd.complete(self.s, args):
                    if cand.lower().startswith(current.lower()):
                        yield Completion(cand, start_position=-len(current))
            return
        if word.startswith("@") and self.s.project:
            for shot in self.s.project.shots:
                tag = "@" + shot.id
                if tag.lower().startswith(word.lower()):
                    yield Completion(tag, start_position=-len(word), display_meta=shot.title)


def _shot_ids(s: Session, args):
    return [x.id for x in s.project.shots] if s.project else []


def _films(s: Session, args):
    return [p.slug for p in s.studio.store.all()]


def _choices(values):
    return lambda s, args: values if len(args) <= 1 else []


# ---- commands: film ------------------------------------------------------------------- #


@command("new", "start a new film (guided, or from an idea)", "Film",
         "[idea] [-t tpl] [-d sec] [-a ratio]",
         complete=lambda s, a: ["-t", "-d", "-a", "--style"] + (list(prompts.TEMPLATES) if a[-2:-1] == ["-t"] else [])
         + (A.ASPECTS if a[-2:-1] == ["-a"] else []))
def cmd_new(s: Session, argv):
    pos, o = opts(argv, {"-t": "template", "--template": "template", "-d": "seconds", "--duration": "seconds",
                         "-a": "aspect", "--aspect": "aspect", "--style": "style"})
    idea = " ".join(pos)
    if not idea:
        if not A.interactive() or s.pt_input is not None:
            raise Abort("give me an idea", '/new "a heist told from the vault\'s point of view"')
        q = A.new_film_questions(s.settings.aspect)
        if q is None:
            return
        s.project = A.start_film(s.studio, **q)
    else:
        try:
            seconds = int(o["seconds"] or 30)
        except ValueError:
            raise Abort("-d takes seconds, e.g. -d 45") from None
        s.project = A.start_film(s.studio, idea, o["template"] or "short-film", seconds, o["aspect"], o["style"] or "")
    A.next_steps(s.project)


@command("open", "switch to another film", "Film", "<film>", aliases=("resume",), complete=_films)
def cmd_open(s: Session, argv):
    if not argv:
        films = _films(s, [])
        if not films:
            raise Abort("no films yet", "describe an idea to start one")
        if not A.interactive() or s.pt_input is not None:
            ui.console.print(ui.project_list(s.studio.store.all()))
            return
        pick = A.choose("Open which film?", films, icon="▶")
        if pick is None:
            return
        argv = [pick]
    s.project = A.find_project(s.studio, argv[0])
    ui.ok(f"opened {s.project.title}")
    if s.project.shots:
        ui.console.print(ui.shot_table(s.project))
    A.next_steps(s.project)


@command("projects", "list your films", "Film", aliases=("films", "list"))
def cmd_projects(s: Session, argv):
    films = s.studio.store.all()
    if not films:
        ui.info(f"no films yet in {s.studio.store.home}")
        return
    ui.console.print(ui.project_list(films))


@command("pitch", "rewrite the treatment", "Film")
def cmd_pitch(s: Session, argv):
    A.do_pitch(s.studio, s.need_project())


@command("plan", "re-plan the shot list", "Film")
def cmd_plan(s: Session, argv):
    A.do_plan(s.studio, s.need_project())
    A.next_steps(s.project)


@command("cast", "draw character reference sheets", "Film", "[-f]", complete=lambda s, a: ["-f"])
def cmd_cast(s: Session, argv):
    _, o = opts(argv, flags={"-f": "force", "--force": "force"})
    A.do_cast(s.studio, s.need_project(), o["force"])


@command("board", "storyboard keyframes", "Film", "[S01 S02…] [-f]", aliases=("storyboard",),
         complete=lambda s, a: _shot_ids(s, a) + ["-f"])
def cmd_board(s: Session, argv):
    pos, o = opts(argv, flags={"-f": "force", "--force": "force"})
    A.do_board(s.studio, s.need_project(), A.split_ids(pos), o["force"])
    A.next_steps(s.project)


@command("shots", "show the shot list", "Film")
def cmd_shots(s: Session, argv):
    p = s.need_project()
    if not p.shots:
        raise Abort("no shot list yet", "/plan")
    ui.console.print(ui.shot_table(p))


@command("show", "storyboard grid, or one shot up close", "Film", "[S03]", complete=_shot_ids)
def cmd_show(s: Session, argv):
    p = s.need_project()
    root = s.studio.store.dir(p)
    if argv:
        try:
            ui.console.print(ui.shot_detail(p, p.shot(argv[0].lstrip("@")), root))
        except KeyError:
            raise Abort(f"no shot {argv[0]}", "/shots lists them") from None
        return
    if p.treatment:
        ui.console.print(ui.treatment_panel(p.treatment, p))
    if p.shots:
        ui.console.print(ui.storyboard(p, root))


@command("templates", "quick-start formats for /new -t", "Film")
def cmd_templates(s: Session, argv):
    ui.console.print(A.templates_table())


# ---- commands: camera ---------------------------------------------------------------- #


@command("shoot", "render shots with LTX-2.5", "Camera", "[S01…] [-b backend] [-q quality] [-f]",
         complete=lambda s, a: (A.BACKENDS if a[-2:-1] == ["-b"] else A.QUALITIES if a[-2:-1] == ["-q"]
                                else _shot_ids(s, a) + ["-b", "-q", "-f"]))
def cmd_shoot(s: Session, argv):
    pos, o = opts(argv, {"-b": "backend", "--backend": "backend", "-q": "quality", "--quality": "quality"},
                  {"-f": "force", "--force": "force"})
    A.do_shoot(s.studio, s.need_project(), A.split_ids(pos), o["backend"], o["quality"], o["force"])
    A.next_steps(s.project)


@command("cut", "assemble the final film", "Camera", "[--music file] [--no-titles]",
         complete=lambda s, a: ["--music", "--no-titles"])
def cmd_cut(s: Session, argv):
    _, o = opts(argv, {"--music": "music", "-m": "music"}, {"--no-titles": "no_titles"})
    music = Path(o["music"]).expanduser() if o["music"] else None
    if music and not music.exists():
        raise Abort(f"no such file: {music}")
    A.do_cut(s.studio, s.need_project(), titles=not o["no_titles"], music=music)
    A.next_steps(s.project)


@command("roll", "make the rest of the film: board → shoot → cut", "Camera", "[-b backend] [-q quality]",
         aliases=("make",), complete=lambda s, a: A.BACKENDS if a[-2:-1] == ["-b"] else
         A.QUALITIES if a[-2:-1] == ["-q"] else ["-b", "-q"])
def cmd_roll(s: Session, argv):
    _, o = opts(argv, {"-b": "backend", "--backend": "backend", "-q": "quality", "--quality": "quality"})
    A.do_roll(s.studio, s.need_project(), o["backend"], o["quality"])
    A.next_steps(s.project)


@command("play", "watch the cut (or a shot)", "Camera", "[S03]", aliases=("watch",), complete=_shot_ids)
def cmd_play(s: Session, argv):
    p = s.need_project()
    root = s.studio.store.dir(p)
    if argv:
        shot = p.shot(argv[0].lstrip("@"))
        if not shot.clip:
            raise Abort(f"{shot.id} isn't shot yet", f"/shoot {shot.id}")
        A.open_file(root / shot.clip)
    elif p.final:
        A.open_file(root / p.final)
    else:
        raise Abort("no cut yet", "/cut")


def _setting(key: str, values: list[str], label: str):
    def fn(s: Session, argv):
        if not argv:
            if A.interactive() and s.pt_input is None:
                pick = A.choose(label, values, default=getattr(s.settings, key))
                if pick is None:
                    return
                argv = [pick]
            else:
                A.config_set(key, None)
                return
        A.config_set(key, argv[0])
        s.reload_settings()

    return fn


command("backend", "how shots are rendered: airstream | ltx-pipelines | animatic", "Camera", "[name]",
        complete=_choices(A.BACKENDS))(_setting("backend", A.BACKENDS, "Render with"))
command("quality", "draft 768p · standard 1024p · hq 1088p two-stage", "Camera", "[level]",
        complete=_choices(A.QUALITIES))(_setting("quality", A.QUALITIES, "Quality"))
command("aspect", "frame for new films: 16:9 9:16 1:1 4:3 2.39:1", "Camera", "[ratio]",
        complete=_choices(A.ASPECTS))(_setting("aspect", A.ASPECTS, "Aspect"))


# ---- commands: setup ----------------------------------------------------------------- #


@command("login", "sign in: ChatGPT (director) or Hugging Face (weights)", "Setup", "[chatgpt|hf]",
         complete=_choices(["chatgpt", "hf"]))
def cmd_login(s: Session, argv):
    A.login(argv[0] if argv else None)
    s.reload_settings()


@command("status", "accounts, director and camera at a glance", "Setup")
def cmd_status(s: Session, argv):
    A.status(s.settings)


@command("model", "pick the director's model", "Setup", "[name | image <name>]",
         complete=lambda s, a: (s._models + ["auto", "image"]) if len(a) <= 1 else s._models)
def cmd_model(s: Session, argv):
    key = "text_model"
    if argv and argv[0] == "image":
        key, argv = "image_model", argv[1:]
    if not argv:
        models = s.studio.llm.models()  # starts the proxy if needed
        s._models = models
        from .llm import rank_image_models, rank_text_models

        ranked = rank_image_models(models) if key == "image_model" else rank_text_models(models)
        current = getattr(s.settings, key) or "auto"
        if not A.interactive() or s.pt_input is not None:
            ui.console.print(Text.assemble(("  current ", "dim"), (current, "accent"), ("   available ", "dim"),
                                           (", ".join(ranked), "ink")))
            return
        pick = A.choose("Director model" if key == "text_model" else "Storyboard model",
                        ["auto (best available)"] + ranked, icon="◆")
        if pick is None:
            return
        argv = ["auto" if pick.startswith("auto") else pick]
    A.config_set(key, argv[0])
    s.reload_settings()


@command("fetch", "download LTX-2.5 + build the layer cache", "Setup", "[--no-hq] [-y]",
         complete=lambda s, a: ["--no-hq", "-y"])
def cmd_fetch(s: Session, argv):
    _, o = opts(argv, flags={"--no-hq": "no_hq", "-y": "yes", "--yes": "yes"})
    A.fetch(hq=not o["no_hq"], yes=o["yes"])


@command("doctor", "check everything Auteur needs", "Setup")
def cmd_doctor(s: Session, argv):
    A.doctor()


@command("proxy", "openai-oauth director proxy", "Setup", "[start|stop|status]",
         complete=_choices(["start", "stop", "status"]))
def cmd_proxy(s: Session, argv):
    A.proxy(argv[0] if argv else "status")


@command("config", "show or change a setting", "Setup", "[key] [value]",
         complete=lambda s, a: list(cfg.Settings.model_fields) if len(a) <= 1 else [])
def cmd_config(s: Session, argv):
    if not argv:
        A.config_show()
        return
    A.config_set(argv[0], " ".join(argv[1:]) if len(argv) > 1 else None)
    s.reload_settings()


# ---- commands: session --------------------------------------------------------------- #


@command("help", "show commands", "Session", aliases=("h", "?"))
def cmd_help(s: Session, argv):
    t = Table(box=None, show_header=False, pad_edge=False, padding=(0, 3))
    t.add_column(no_wrap=True)
    t.add_column(style="ink")
    for group in GROUPS:
        t.add_row(Text(group.upper(), style="title"), "")
        for c in COMMANDS.values():
            if c.group == group:
                # Text, not markup: usages like [name] would be eaten as style tags
                t.add_row(Text.assemble(("  /" + c.name, "accent"), (" " + c.usage if c.usage else "", "dim")),
                          c.help)
        t.add_row("", "")
    ui.console.print(t)
    ui.console.print(Text.assemble(
        ("  Anything else you type goes to the director: ", "dim"),
        ("an idea", "ink"), (" starts a film, ", "dim"), ("a note", "ink"), (" revises it. ", "dim"),
        ("@S03", "accent"), (" points at a shot.", "dim")))


@command("clear", "clear the screen", "Session")
def cmd_clear(s: Session, argv):
    ui.console.clear()
    if s.project:
        ui.console.print(Text.assemble(("  ◆ ", "accent"), (s.project.title, "title")))


@command("exit", "leave Auteur", "Session", aliases=("quit", "q"))
def cmd_exit(s: Session, argv):
    s.running = False

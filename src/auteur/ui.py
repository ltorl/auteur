"""Terminal look & feel: banner, panels, storyboard grid, live render dashboard."""

from __future__ import annotations

import contextlib
import itertools
import time
from collections.abc import Iterator
from pathlib import Path

from rich import box
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from . import __version__
from .schemas import Project, Shot, Treatment

AMBER = "#FFB000"
EMBER = "#FF6A3D"
ROSE = "#FF3D7F"
TEAL = "#3DDBD9"
INK = "#ECE6DA"
SMOKE = "#8A857C"
NIGHT = "#1B1A1F"

theme = Theme(
    {
        "accent": f"bold {AMBER}",
        "ember": EMBER,
        "teal": TEAL,
        "ink": INK,
        "dim": SMOKE,
        "ok": "bold #7BD88F",
        "warn": "bold #FFD166",
        "err": f"bold {ROSE}",
        "title": f"bold {INK}",
    }
)
console = Console(theme=theme, highlight=False)
MAX_WIDTH = 120


def fit() -> None:
    """Cap output width: on very wide terminals tables and rules get unreadable."""
    import shutil

    console.width = min(shutil.get_terminal_size((100, 40)).columns, MAX_WIDTH)

LETTERS = {
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "U": ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
}

STATUS = {
    "planned": ("○", SMOKE, "planned"),
    "boarded": ("◐", AMBER, "boarded"),
    "shot": ("●", "#7BD88F", "shot"),
    "failed": ("✕", ROSE, "failed"),
}


def _lerp(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def gradient(text: str, stops=(AMBER, EMBER, ROSE)) -> Text:
    out = Text()
    n = max(len(text) - 1, 1)
    for i, ch in enumerate(text):
        t = i / n * (len(stops) - 1)
        k = min(int(t), len(stops) - 2)
        out.append(ch, style=f"bold {_lerp(stops[k], stops[k + 1], t - k)}")
    return out


def wordmark() -> Text:
    rows = ["".join(LETTERS[c][r] for c in "AUTEUR") for r in range(6)]
    out = Text()
    for row in rows:
        out.append_text(gradient(row))
        out.append("\n")
    out.rstrip()
    return out


def film_strip(width: int) -> Text:
    return Text(("▆ " * (width // 2 + 1))[:width], style=SMOKE)


def banner(subtitle: str | None = None) -> None:
    mark = wordmark()
    w = max(len(line) for line in mark.plain.splitlines())
    tag = Text.assemble(("an AI film director in your terminal", "ink"), ("  ·  ", "dim"), (f"v{__version__}", "dim"))
    strip = film_strip(w + 2)
    sub = [Text(subtitle, style="dim")] if subtitle else []
    body = Group(strip, Text(""), mark, Text(""), tag, *sub, Text(""), strip)
    console.print(Padding(body, (1, 0, 0, 0)))


def section(title: str, icon: str = "◆") -> None:
    console.print()
    console.print(Rule(Text.assemble((f" {icon} ", "accent"), (title.upper(), "title"), " "), style=SMOKE,
                       align="left"))


def ok(msg: str) -> None:
    console.print(Text.assemble(("  ✓ ", "ok"), (msg, "ink")))


def warn(msg: str) -> None:
    console.print(Text.assemble(("  ! ", "warn"), (msg, "ink")))


def err(msg: str, hint: str = "") -> None:
    console.print(Text.assemble(("  ✕ ", "err"), (msg, "ink")))
    if hint:
        console.print(Text.assemble(("    → ", "dim"), (hint, "teal")))


def info(msg: str) -> None:
    console.print(Text.assemble(("  · ", "dim"), (msg, "dim")))


def tilde(path: Path | str) -> str:
    p, home = str(path), str(Path.home())
    return "~" + p[len(home):] if p.startswith(home) else p


def path_link(path: Path) -> Text:
    return Text(str(path), style=f"underline {TEAL} link file://{path}")


# ---- images in the terminal ------------------------------------------------ #


def thumbnail(path: Path | None, cols: int = 28, rows: int = 8) -> Text:
    """Render an image with ▀ half-blocks (2 pixels per cell, truecolor)."""
    if not path or not Path(path).exists():
        t = Text()
        for r in range(rows):
            line = "░" * cols if r not in (rows // 2,) else ("no frame".center(cols, "░"))
            t.append(line + ("\n" if r < rows - 1 else ""), style=f"{SMOKE} on {NIGHT}")
        return t
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.fit(im.convert("RGB"), (cols, rows * 2), Image.LANCZOS)
        px = im.load()
    t = Text()
    for y in range(0, rows * 2, 2):
        for x in range(cols):
            top, bot = px[x, y], px[x, y + 1]
            t.append("▀", style=f"#{top[0]:02x}{top[1]:02x}{top[2]:02x} on #{bot[0]:02x}{bot[1]:02x}{bot[2]:02x}")
        if y < rows * 2 - 2:
            t.append("\n")
    return t


# ---- film views -------------------------------------------------------------- #


def treatment_panel(t: Treatment, p: Project) -> Panel:
    head = Text.assemble((t.title.upper(), f"bold {AMBER}"), "\n", (t.logline, f"italic {INK}"))
    meta = Text.assemble(
        (t.genre or p.template, "teal"), ("  ·  ", "dim"), (t.tone, "ink"), ("  ·  ", "dim"),
        (f"{p.aspect}  ~{p.target_seconds}s", "dim"),
    )
    style = Table.grid(padding=(0, 2))
    style.add_column(style="dim", no_wrap=True)
    style.add_column(style="ink")
    for label, val in [("look", t.style.visual_style), ("palette", t.style.palette), ("light", t.style.lighting),
                       ("camera", t.style.camera_language), ("texture", t.style.texture), ("sound", t.style.sound)]:
        if val:
            style.add_row(label, val)
    cast = Table(box=box.SIMPLE_HEAD, show_edge=False, header_style="dim", pad_edge=False, expand=True)
    cast.add_column("cast", style=f"bold {INK}", no_wrap=True)
    cast.add_column("", style="ink", ratio=1)
    for c in t.characters:
        cast.add_row(c.name, Text.assemble((c.role + " — " if c.role else "", "teal"), (c.look, "ink")))
    for loc in t.locations:
        cast.add_row(Text(loc.name, style="dim"), Text(loc.look, style="dim"))
    body = Group(head, Text(""), meta, Text(""), Text(t.synopsis, style="ink"), Text(""), style, Text(""), cast)
    return Panel(body, box=box.HEAVY_HEAD, border_style=SMOKE, padding=(1, 2),
                 title=Text(" TREATMENT ", style=f"bold {NIGHT} on {AMBER}"), title_align="left")


def status_text(s: Shot) -> Text:
    icon, color, label = STATUS.get(s.status, STATUS["planned"])
    return Text(f"{icon} {label}", style=color)


def shot_table(p: Project) -> Table:
    t = Table(box=box.SIMPLE_HEAVY, header_style=f"bold {SMOKE}", border_style=SMOKE, expand=True,
              title=Text.assemble((p.title, "title"), ("   ", ""), (f"{len(p.shots)} shots · {p.runtime:.0f}s", "dim")),
              title_justify="left", pad_edge=False)
    t.add_column("#", style="accent", no_wrap=True)
    t.add_column("shot", style="ink", ratio=3)
    t.add_column("size", style="teal", no_wrap=True)
    t.add_column("camera", style="ink", ratio=2)
    t.add_column("cast", style="dim", ratio=1)
    t.add_column("sec", justify="right", style="ink", no_wrap=True)
    t.add_column("", no_wrap=True)
    scene = None
    for s in p.shots:
        if scene is not None and s.scene != scene:
            t.add_section()
        scene = s.scene
        title = Text(s.title, style="ink")
        if s.dialogue:
            title.append(f'\n“{s.dialogue[0].line}”', style=f"italic {SMOKE}")
        t.add_row(s.id, title, s.shot_size, s.camera, ", ".join(s.characters), f"{s.duration:.1f}", status_text(s))
    return t


def shot_card(p: Project, s: Shot, root: Path, cols: int = 30) -> Panel:
    img = thumbnail(root / s.keyframe if s.keyframe else None, cols=cols, rows=cols * 9 // 32)
    cap = Text.assemble((f"{s.id} ", "accent"), (s.shot_size, "teal"), ("  ", ""), (f"{s.duration:.0f}s", "dim"))
    cam = Text(s.camera, style="ink", overflow="ellipsis", no_wrap=True)
    return Panel(Group(img, cap, cam), box=box.ROUNDED, border_style=STATUS[s.status][1] if s.status != "planned" else SMOKE,
                 padding=(0, 1), width=cols + 4, subtitle=status_text(s), subtitle_align="right")


def storyboard(p: Project, root: Path, cols: int = 30) -> Columns:
    return Columns([shot_card(p, s, root, cols) for s in p.shots], padding=(0, 1))


def shot_detail(p: Project, s: Shot, root: Path) -> Panel:
    img = thumbnail(root / s.keyframe if s.keyframe else None, cols=64, rows=18)
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim", no_wrap=True)
    meta.add_column(style="ink")
    meta.add_row("framing", f"{s.shot_size}  {s.lens}".strip())
    meta.add_row("camera", s.camera)
    meta.add_row("cast", ", ".join(s.characters) or "—")
    meta.add_row("where", s.location or "—")
    meta.add_row("action", s.action)
    for line in s.dialogue:
        meta.add_row("line", Text(f'{line.speaker}: “{line.line}”', style=f"italic {INK}"))
    if s.sound:
        meta.add_row("sound", s.sound)
    meta.add_row("seed", str(s.seed))
    if s.clip:
        meta.add_row("clip", path_link(root / s.clip))
    if s.error:
        meta.add_row("error", Text(s.error[:300], style="err"))
    prompt = Panel(Text(s.motion_prompt, style="ink"), title=Text("LTX prompt", style="dim"), title_align="left",
                   border_style=SMOKE, box=box.MINIMAL)
    return Panel(Group(img, Text(""), meta, prompt), title=Text.assemble((f" {s.id} ", f"bold {NIGHT} on {AMBER}"),
                 (f" {s.title} ", "title")), title_align="left", border_style=SMOKE, box=box.HEAVY_HEAD,
                 subtitle=status_text(s), subtitle_align="right")


def project_list(projects: list[Project]) -> Table:
    t = Table(box=box.SIMPLE_HEAD, header_style=f"bold {SMOKE}", expand=True, pad_edge=False)
    t.add_column("project", style="accent", no_wrap=True)
    t.add_column("title", style="ink", ratio=2)
    t.add_column("shots", justify="right", style="ink")
    t.add_column("progress", ratio=1)
    t.add_column("updated", style="dim", no_wrap=True)
    for p in projects:
        n = len(p.shots) or 1
        shot = sum(s.status == "shot" for s in p.shots)
        boarded = sum(s.status in ("boarded", "shot") for s in p.shots)
        bar = Text("█" * shot, style="#7BD88F") + Text("█" * (boarded - shot), style=AMBER) + Text(
            "░" * (n - boarded), style=SMOKE)
        if p.final:
            bar.append("  ★ cut", style="accent")
        t.add_row(p.slug, p.title, str(len(p.shots)), bar if p.shots else Text("idea", style="dim"),
                  p.updated.strftime("%b %d %H:%M"))
    return t


# ---- host: where live views and questions go ---------------------------------- #
#
# The full-screen app installs itself as the host so spinners, progress bars and
# questions render inside it. Without a host (scripts, pipes) we fall back to
# Rich's Live and questionary.


class Host:
    def live_push(self, get) -> None: ...
    def live_pop(self, get) -> None: ...
    def select(self, question: str, choices: list[str], default: str | None) -> str | None: ...
    def text(self, question: str, secret: bool) -> str | None: ...
    def external(self, fn): ...
    def clear(self) -> None: ...


host: Host | None = None


@contextlib.contextmanager
def live(get, transient: bool = True, fps: int = 10) -> Iterator:
    """Show ``get()`` live while the block runs; keep the last frame unless transient."""
    if host is not None:
        host.live_push(get)
        try:
            yield
        finally:
            host.live_pop(get)
            if not transient:
                console.print(get())
        return
    with Live(get(), console=console, refresh_per_second=fps, transient=transient, get_renderable=get):
        yield


class _Status:
    def __init__(self, text: str):
        self.text = text
        self.frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

    def update(self, text: str) -> None:
        self.text = text

    def view(self) -> Text:
        return Text.assemble((f"  {next(self.frames)} ", "accent"), Text.from_markup(self.text, style="ink"))


@contextlib.contextmanager
def status(text: str) -> Iterator[_Status]:
    """A one-line spinner; ``.update()`` changes its text (Rich markup allowed)."""
    st = _Status(text)
    with live(st.view, fps=12):
        yield st


def select(question: str, choices: list[str], default: str | None = None) -> str | None:
    if host is not None:
        return host.select(question, choices, default)
    import questionary

    return questionary.select(question, choices=choices, default=default, style=_qstyle(), qmark="▸").ask()


def ask_text(question: str, secret: bool = False) -> str | None:
    if host is not None:
        return host.text(question, secret)
    import questionary

    fn = questionary.password if secret else questionary.text
    return fn(question, style=_qstyle(), qmark="✎").ask()


def confirm(question: str, default: bool = True) -> bool:
    yes, no = "Yes", "No"
    return select(question, [yes, no] if default else [no, yes], yes if default else no) == yes


def external(fn):
    """Run ``fn`` with the real terminal (interactive subprocesses)."""
    return host.external(fn) if host is not None else fn()


_QSTYLE = None


def _qstyle():
    global _QSTYLE
    if _QSTYLE is None:
        from questionary import Style

        _QSTYLE = Style([("qmark", f"fg:{AMBER} bold"), ("question", "bold"), ("answer", f"fg:{AMBER} bold"),
                         ("pointer", f"fg:{AMBER} bold"), ("highlighted", f"fg:{AMBER} bold"),
                         ("selected", f"fg:{TEAL}"), ("instruction", f"fg:{SMOKE}")])
    return _QSTYLE


# ---- live feedback ------------------------------------------------------------ #

REEL = ["◐", "◓", "◑", "◒"]


@contextlib.contextmanager
def thinking(label: str) -> Iterator:
    """Spinner + live token count + a peek at what the director is writing."""
    state = {"n": 0, "tail": "", "t0": time.time()}
    frames = itertools.cycle(REEL)

    def view() -> RenderableType:
        tail = " ".join(state["tail"].split())[-90:]
        line = Text.assemble((f"  {next(frames)} ", "accent"), (label, "ink"),
                             (f"   {state['n']:,} chars · {time.time() - state['t0']:.0f}s", "dim"))
        return Group(line, Text(f"    {tail}", style=f"italic {SMOKE}", no_wrap=True, overflow="ellipsis"))

    def progress(n: int, tail: str) -> None:
        state["n"], state["tail"] = n, tail

    with live(view, fps=12):
        yield progress
    ok(f"{label} — {time.time() - state['t0']:.0f}s")


class RenderView:
    """Live dashboard for the camera department."""

    def __init__(self, total_shots: int, backend: str):
        self.progress = Progress(
            SpinnerColumn("dots", style=AMBER),
            TextColumn("[ink]{task.description}"),
            BarColumn(bar_width=None, style=SMOKE, complete_style=AMBER, finished_style="#7BD88F"),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[extra]}"),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        )
        self.shots = self.progress.add_task("shots", total=total_shots, extra=backend)
        self.steps = self.progress.add_task("denoise", total=8, extra="", visible=False)
        self.blocks = self.progress.add_task("stream", total=1, extra="", visible=False)
        self.phase = Text("", style="dim")
        self._cm = live(self._view, transient=False)

    def _view(self):
        return Panel(Group(self.progress, self.phase), border_style=SMOKE, box=box.ROUNDED,
                     title=Text(" CAMERA ROLLING ", style=f"bold {NIGHT} on {ROSE}"), title_align="left")

    def __enter__(self):
        self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def __call__(self, e) -> None:
        if e.kind == "phase":
            self.phase = Text.assemble(("  ▸ ", "accent"), (e.text, "ink"))
            if e.shot:
                self.progress.update(self.steps, description=f"{e.shot} denoise")
        elif e.kind == "step":
            self.progress.update(self.steps, total=e.n, completed=e.i, visible=True)
        elif e.kind == "unit":
            self.progress.update(self.blocks, description=f"{e.text} layers", total=e.n, completed=e.i,
                                 visible=True, extra="streaming from disk")
        elif e.kind == "log":
            self.phase = Text.assemble(("  · ", "dim"), (escape(e.text), "dim"))
        elif e.kind == "done":
            self.progress.advance(self.shots)
            finished = self.progress.tasks[self.shots].finished
            self.progress.update(self.steps, completed=0, visible=not finished)
            self.progress.update(self.blocks, visible=not finished)
            if finished:
                self.phase = Text.assemble(("  ▸ ", "accent"), ("that's a wrap", "ink"))

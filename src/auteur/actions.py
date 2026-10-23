"""Everything a command does, shared by the interactive session and subcommands.

Functions print through :mod:`auteur.ui` and raise :class:`Abort` for failures
the user can act on, so callers only decide how to recover.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config as cfg
from . import prompts, ui
from .llm import ProxyError, codex_auth_file, proxy_up
from .schemas import Project
from .studio import Studio

ASPECTS = ["16:9", "9:16", "1:1", "4:3", "2.39:1"]
BACKENDS = ["airstream", "ltx-pipelines", "animatic"]
QUALITIES = ["draft", "standard", "hq"]
HF_REPO = "Lightricks/LTX-2.5-Diffusers"


class Abort(Exception):
    """A failure with something the user can do about it."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


# ---- small helpers ------------------------------------------------------------ #


def studio() -> Studio:
    return Studio(cfg.load())


def find_project(s: Studio, ref: str | None) -> Project:
    try:
        p = s.store.load(ref) if ref else s.store.latest()
    except FileNotFoundError as e:
        raise Abort(str(e), "/projects lists your films") from None
    if p is None:
        raise Abort("no film open", "describe an idea to start one, or /new")
    return p


def split_ids(shots: str | list[str] | None) -> list[str] | None:
    if not shots:
        return None
    raw = ",".join(shots) if isinstance(shots, list) else shots
    return [x.strip().upper() for x in raw.replace(" ", ",").split(",") if x.strip()] or None


def open_file(path: Path) -> None:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener):
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def choose(question: str, choices: list[str], default: str | None = None, icon: str = "▸") -> str | None:
    return ui.select(question, choices, default)


def confirm(question: str, default: bool = True) -> bool:
    return ui.confirm(question, default)


def text(question: str, icon: str = "✎", secret: bool = False) -> str | None:
    return ui.ask_text(question, secret)


def next_steps(p: Project, storyboard: bool = True) -> None:
    """Suggest what to do next, given where the film is."""
    if not p.shots:
        tip = "/plan to break it into shots"
    elif storyboard and any(s.status == "planned" for s in p.shots):
        tip = "/board to storyboard  ·  /roll to make the whole film  ·  or type a note"
    elif any(s.status != "shot" for s in p.shots):
        tip = "/shoot to roll camera  ·  /shoot -b animatic for a quick preview"
    elif not p.final:
        tip = "/cut to assemble the film"
    else:
        tip = "/play to watch  ·  type a note to keep refining"
    ui.console.print(Text.assemble(("  next  ", "dim"), (tip, "teal")))


def storyboard_on(s: Studio) -> bool:
    return s.settings.storyboard and not s.images_blocked


def images_refused(reason: str) -> None:
    """One clear explanation instead of an error per shot."""
    ui.console.print(Panel(
        Text.assemble(
            ("ChatGPT refused image generation ", "ink"), (f"({reason})", "err"), (".\n", "ink"),
            ("Your plan may not include images through the Codex sign-in. ", "dim"),
            ("No problem: LTX-2.5 will shoot each shot ", "ink"), ("text-to-video", "accent"),
            (" from its written prompt.\n\n", "ink"),
            ("/config storyboard false", "teal"), ("  stop trying   ", "dim"),
            ("/model image", "teal"), ("  try another image model", "dim"),
        ),
        border_style=ui.AMBER, box=box.ROUNDED, padding=(0, 2), title=Text(" no storyboard ", style="warn"),
        title_align="left", width=min(ui.console.width, 100)))


def preflight(s: Studio, backend: str | None = None) -> str:
    """Make sure the camera can actually roll before spending director/image calls."""
    from . import weights as W

    backend = backend or s.settings.backend
    if backend not in BACKENDS:
        raise Abort(f"unknown backend '{backend}'", " | ".join(BACKENDS))
    if not shutil.which("ffmpeg"):
        raise Abort("ffmpeg isn't installed", "brew install ffmpeg")
    if backend == "airstream":
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            raise Abort("the local LTX engine isn't installed (lite install)",
                        "reinstall without AUTEUR_LITE=1, or /backend animatic") from None
        dest = Path(s.settings.weights).expanduser()
        if not W.present(dest):
            prog = W.progress(dest)
            if prog and prog.done_bytes:
                what = (f"LTX-2.5 is only {prog.fraction:.0%} downloaded "
                        f"({prog.done_bytes / 1e9:.1f} of {prog.total_bytes / 1e9:.0f} GB)")
            else:
                what = "LTX-2.5 isn't downloaded yet"
            raise Abort(what, "/fetch to download (it resumes where it stopped), or /backend animatic to preview")
    elif backend == "ltx-pipelines":
        if not (s.settings.ltx2_repo and Path(s.settings.ltx2_repo).expanduser().exists()):
            raise Abort("ltx-pipelines needs a Lightricks/LTX-2 checkout", "/config ltx2_repo /path/to/LTX-2")
        if not (s.settings.ltx_checkpoints and Path(s.settings.ltx_checkpoints).expanduser().exists()):
            raise Abort("ltx-pipelines needs the LTX-2.5 checkpoints", "/config ltx_checkpoints /path/to/LTX-2.5")
    return backend


# ---- production ------------------------------------------------------------------ #


def do_pitch(s: Studio, p: Project) -> None:
    ui.section("Treatment", "✦")
    with ui.thinking(f"Writing the treatment with {s.llm.text_model}") as prog:
        s.pitch(p, prog)
    ui.console.print(ui.treatment_panel(p.treatment, p))


def do_plan(s: Studio, p: Project) -> None:
    ui.section("Shot list", "▤")
    with ui.thinking("Breaking it down into shots") as prog:
        s.plan(p, prog)
    ui.console.print(ui.shot_table(p))


def _art(s: Studio, label: str, run, total: int) -> None:
    """Run an image batch with live count; collapse refusals into one message."""
    from .studio import ImagesUnavailable

    with ui.status(f"[ink]{label}…") as st:
        n = [0]

        def done(name, path, err):
            n[0] += 1
            st.update(f"[ink]{label}… {n[0]}/{total}")
            if path:
                ui.ok(f"{name}  →  {path.name}")
            elif not isinstance(err, ImagesUnavailable):
                ui.err(f"{name}: {err}")

        run(done)
    if s.images_blocked:
        images_refused(s.images_blocked)


def do_cast(s: Studio, p: Project, force: bool = False) -> None:
    if not p.treatment or not p.treatment.characters or not storyboard_on(s):
        return
    ui.section("Casting", "☺")
    _art(s, f"Drawing character sheets with {s.llm.image_model}",
         lambda done: s.cast(p, force=force, on_item=done), len(p.treatment.characters))


def do_board(s: Studio, p: Project, ids=None, force: bool = False) -> None:
    if not p.shots:
        raise Abort("no shot list yet", "/plan first")
    if not s.settings.storyboard:
        raise Abort("storyboards are switched off", "/config storyboard true  (LTX shoots text-to-video meanwhile)")
    if s.images_blocked:
        images_refused(s.images_blocked)
        return
    if not p.cast_sheets and p.treatment and p.treatment.characters:
        do_cast(s, p)
        if s.images_blocked:
            return
    ui.section("Storyboard", "▦")
    todo = [x for x in p.shots if ids is None or x.id.upper() in ids]
    _art(s, f"Boarding {len(todo)} shots", lambda done: s.board(p, ids, force=force, on_item=done), len(todo))
    if any(x.keyframe for x in p.shots):
        ui.console.print(ui.storyboard(p, s.store.dir(p)))


def do_shoot(s: Studio, p: Project, ids=None, backend=None, quality=None, force=False) -> None:
    if not p.shots:
        raise Abort("no shot list yet", "/plan first")
    quality = quality or s.settings.quality
    backend = preflight(s, backend)
    if quality not in QUALITIES:
        raise Abort(f"unknown quality '{quality}'", " | ".join(QUALITIES))
    jobs = s.jobs(p, ids, quality, force)
    ui.section("Shooting", "●")
    if not jobs:
        ui.info("every requested shot is already in the can (add -f to reshoot)")
        return
    w, h = jobs[0].width, jobs[0].height
    n = f"{len(jobs)} shot{'s' if len(jobs) != 1 else ''}"
    t2v = sum(1 for j in jobs if not (j.image and j.image.exists()))
    if t2v and backend != "animatic":
        ui.info(f"{t2v} shot{'s' if t2v != 1 else ''} without a keyframe will be filmed text-to-video")
    ui.info(f"{n} · {backend} · {quality} {w}×{h} · {sum(j.seconds for j in jobs):.0f}s of footage")
    if backend == "airstream":
        ui.info("LTX-2.5 22B streams from disk one block at a time — slow but fits in laptop memory")
    with ui.RenderView(len(jobs), backend) as view:
        s.shoot(p, view, ids=ids, backend=backend, quality=quality, force=force)
    ui.ok(f"{n} in the can")


def do_cut(s: Studio, p: Project, titles=True, music=None, open_after=False) -> Path:
    ui.section("Final cut", "✂")
    with ui.status("[ink]Cutting…") as st:
        out = s.cut(p, titles=titles, music=music, on_step=lambda m: st.update(f"[ink]{m}…"))
    ui.ok("Premiere ready")
    ui.console.print(Text.assemble(("    ", ""), ui.path_link(out)))
    if open_after:
        open_file(out)
    return out


def do_revise(s: Studio, p: Project, note: str) -> None:
    with ui.thinking("Director is considering your note") as prog:
        rev = s.revise(p, note, prog)
    ui.console.print(Text.assemble(("  director › ", "accent"), (rev.reply, "ink")))
    if rev.treatment is not None:
        ui.console.print(ui.treatment_panel(p.treatment, p))
    if rev.shots is not None or rev.treatment is not None:
        ui.console.print(ui.shot_table(p))


def do_roll(s: Studio, p: Project, backend=None, quality=None, open_after=False) -> None:
    """Everything that's left: plan → cast → board → shoot → cut."""
    backend = preflight(s, backend)  # fail before spending any director/image calls
    if not p.treatment:
        do_pitch(s, p)
    if not p.shots:
        do_plan(s, p)
    if storyboard_on(s) and any(x.status == "planned" for x in p.shots):
        do_board(s, p)
    do_shoot(s, p, backend=backend, quality=quality)
    do_cut(s, p, open_after=open_after)


def start_film(s: Studio, idea: str, template: str = "short-film", seconds: int = 30,
               aspect: str | None = None, style: str = "") -> Project:
    aspect = aspect or s.settings.aspect
    if template not in prompts.TEMPLATES:
        raise Abort(f"unknown template '{template}'", "/templates lists them")
    if aspect not in ASPECTS:
        raise Abort(f"unknown aspect '{aspect}'", " | ".join(ASPECTS))
    s.llm.ensure()
    p = s.create(idea, template, seconds, aspect, style)
    ui.info(f"new film  {s.store.dir(p)}")
    do_pitch(s, p)
    do_plan(s, p)
    return p


def new_film_questions(defaults_aspect: str) -> dict | None:
    """The guided /new flow. Returns kwargs for :func:`start_film`."""
    idea = text("What's the idea?", icon="🎬")
    if not idea:
        return None
    labels = {f"{v['label']:<14} {v['brief'][:56]}…": k for k, v in prompts.TEMPLATES.items()}
    tpl = choose("Format", list(labels), icon="▤")
    if tpl is None:
        return None
    seconds = choose("Runtime (seconds)", ["15", "30", "45", "60", "90"], default="30", icon="⏱")
    aspect = choose("Aspect", ASPECTS, default=defaults_aspect, icon="▭")
    if seconds is None or aspect is None:
        return None
    style = text("Style notes (optional)", icon="🎨") or ""
    return {"idea": idea, "template": labels[tpl], "seconds": int(seconds), "aspect": aspect, "style": style}


def templates_table() -> Table:
    t = Table(box=None, pad_edge=False, show_header=False)
    t.add_column(style="accent", no_wrap=True)
    t.add_column(style="title", no_wrap=True)
    t.add_column(style="dim")
    for k, v in prompts.TEMPLATES.items():
        t.add_row(k, v["label"], v["brief"])
    return t


# ---- accounts ---------------------------------------------------------------------- #


def hf_user() -> str | None:
    try:
        from huggingface_hub import get_token, whoami

        if not get_token():
            return None
        return whoami().get("name")
    except Exception:  # noqa: BLE001 — offline / bad token / not installed
        return None


def hf_access() -> bool | None:
    """True if this account may download LTX-2.5; None if unknown."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError

        HfApi().auth_check(HF_REPO)
        return True
    except GatedRepoError:
        return False
    except Exception:  # noqa: BLE001
        return None


def login_chatgpt() -> None:
    from .llm import login as oauth_login

    ui.section("Sign in · ChatGPT", "🔑")
    ui.info("opens your browser — credentials are stored in ~/.codex (shared with the Codex CLI)")
    try:
        code = ui.external(oauth_login)
    except ProxyError as e:
        raise Abort(str(e), e.hint) from None
    if code != 0 or not codex_auth_file().exists():
        raise Abort("ChatGPT sign-in did not complete", "try /login again")
    ui.ok("signed in to ChatGPT — your director is ready")


def login_hf(token: str | None = None) -> None:
    ui.section("Sign in · Hugging Face", "🤗")
    ui.info("needed to download LTX-2.5 · create a Read token at https://huggingface.co/settings/tokens")
    token = token or text("Paste your Hugging Face token", icon="🔑", secret=True)
    if not token:
        raise Abort("no token given")
    from huggingface_hub import login

    try:
        login(token=token.strip(), add_to_git_credential=False)
    except Exception as e:  # noqa: BLE001
        raise Abort(f"Hugging Face rejected the token: {e}") from None
    ui.ok(f"signed in to Hugging Face as {hf_user() or 'you'}")
    access = hf_access()
    if access is False:
        ui.warn(f"accept the LTX-2.5 license first: https://huggingface.co/{HF_REPO}")
    elif access:
        ui.ok("LTX-2.5 access granted — /fetch to download")


def login(which: str | None = None) -> None:
    options = {"ChatGPT  — the director (writing + storyboards)": "chatgpt",
               "Hugging Face  — LTX-2.5 weights download": "hf"}
    if which is None:
        if not interactive():
            which = "chatgpt"
        else:
            pick = choose("Sign in to", list(options), icon="🔑")
            if pick is None:
                return
            which = options[pick]
    which = {"openai": "chatgpt", "gpt": "chatgpt", "huggingface": "hf", "hugging-face": "hf"}.get(which, which)
    if which == "chatgpt":
        login_chatgpt()
    elif which == "hf":
        login_hf()
    else:
        raise Abort(f"unknown account '{which}'", "/login chatgpt  or  /login hf")


def _weights_row(dest: Path) -> tuple[bool | None, str, str]:
    from . import weights as W

    if W.present(dest):
        return True, "LTX-2.5", ui.tilde(dest)
    prog = W.progress(dest)
    if prog and prog.done_bytes:
        return None, "LTX-2.5", (f"{prog.fraction:.0%} downloaded ({prog.done_bytes / 1e9:.1f}/"
                                 f"{prog.total_bytes / 1e9:.0f} GB) — /fetch resumes")
    return None, "LTX-2.5", "/fetch"


def status(settings: cfg.Settings) -> None:
    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(no_wrap=True)
    t.add_column(style="ink", no_wrap=True)
    t.add_column(style="dim")

    def row(good, name, detail):
        mark = Text("  ●", style="ok") if good else Text("  ●", style="err") if good is False else Text("  ○", style="warn")
        t.add_row(mark, name, detail)

    signed = codex_auth_file().exists()
    row(signed, "ChatGPT", "signed in" if signed else "/login chatgpt")
    up = proxy_up(settings.base_url)
    row(up or None, "director", settings.base_url if up else "offline — starts on first use")
    user = hf_user()
    row(bool(user) or None, "Hugging Face", user or "/login hf (only to download weights)")

    row(*_weights_row(Path(settings.weights).expanduser()))
    row(True, "camera", f"{settings.backend} · {settings.quality} · {settings.aspect}")
    ui.console.print(t)


# ---- proxy -------------------------------------------------------------------------- #


def proxy(action: str = "status") -> None:
    from .llm import start_proxy, stop_proxy

    s = cfg.load()
    if action == "start":
        if proxy_up(s.base_url):
            ui.ok(f"director already online at {s.base_url}")
            return
        if not codex_auth_file().exists():
            raise Abort("not signed in to ChatGPT", "/login")
        try:
            with ui.status("[ink]Starting openai-oauth…"):
                start_proxy(s.base_url)
        except ProxyError as e:
            raise Abort(str(e), e.hint) from None
        ui.ok(f"director online at {s.base_url}")
    elif action == "stop":
        stop_proxy()
        ui.ok("director proxy stopped")
    elif action == "status":
        if proxy_up(s.base_url):
            ui.ok(f"online at {s.base_url}")
        else:
            ui.warn(f"offline ({s.base_url}) — starts automatically when needed")
    else:
        raise Abort(f"unknown proxy action '{action}'", "start | stop | status")


# ---- weights ------------------------------------------------------------------------ #


def fetch(hq: bool = True, compress: str | None = None, yes: bool = False) -> None:
    from rich.progress import (BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn,
                               TransferSpeedColumn)

    from . import weights as W

    s = cfg.load()
    dest = Path(s.weights).expanduser()
    compress = compress or s.compress
    ui.section("LTX-2.5 weights", "⬇")
    if W.present(dest):
        ui.ok(f"already downloaded  {ui.tilde(dest)}")
    else:
        if W.token() is None:
            raise Abort("not signed in to Hugging Face",
                        f"accept the license at https://huggingface.co/{HF_REPO}, then /login hf")
        try:
            with ui.status("[ink]Reading the model index…"):
                files = W.plan(dest, hq=hq)
                sizes = W.sizes(files)
        except W.GatedError:
            raise Abort("access denied by Hugging Face",
                        f"click 'Agree and Access' on https://huggingface.co/{HF_REPO} with this account") from None
        W.save_plan(dest, sizes)
        prog = W.progress(dest)
        total = sum(sizes.values())
        anchor = next(p for p in [dest, *dest.parents] if p.exists())
        free = shutil.disk_usage(anchor).free
        have = prog.done_bytes if prog else 0
        ui.info(f"{len(files)} files · {total / 1e9:.1f} GB → {ui.tilde(dest)}   ({free / 1e9:.0f} GB free)")
        if have:
            ui.info(f"resuming — {have / 1e9:.1f} GB already here")
        if compress == "fp8":
            ui.info("then an fp8 layer cache of the streamed components (~35 GB)")
        if free < (total - have) * 1.05 + (35e9 if compress == "fp8" else 0):
            ui.warn("that may not fit on this disk")
        if not yes and interactive() and not confirm("Download?"):
            return
        bar = Progress(
            TextColumn("  [accent]⬇[/] [ink]LTX-2.5"), BarColumn(bar_width=None, style=ui.SMOKE, complete_style=ui.AMBER),
            TextColumn("[ink]{task.percentage:>3.0f}%"), DownloadColumn(), TransferSpeedColumn(),
            TextColumn("[dim]eta"), TimeRemainingColumn(), TextColumn("[dim]{task.fields[files]}"),
            console=ui.console, expand=True,
        )
        with ui.live(lambda: bar, fps=4):
            task = bar.add_task("weights", total=total, completed=have, files="")
            W.download(dest, files, on_progress=lambda p: bar.update(
                task, completed=p.done_bytes, files=f"{p.done_files}/{p.total_files} files"))
        ui.ok("weights ready  (safe to interrupt and re-run /fetch at any point — it resumes)")
    from .render.ltx import verify

    with ui.status("[ink]Checking the checkpoint against the model…"):
        gaps = {c: m for c, m in verify(dest).items() if m}
    if gaps:
        detail = "; ".join(f"{c}: {len(m)} unmatched (e.g. {m[0]})" for c, m in gaps.items())
        raise Abort(f"checkpoint doesn't match the model — {detail}",
                    "your diffusers/transformers may be older than the checkpoint; reinstall Auteur to update")
    ui.ok("checkpoint keys match the model")
    if compress == "fp8":
        from .render.ltx import build_caches

        with ui.status("[ink]Splitting layers…") as st:
            build_caches(dest, Path(s.layer_cache).expanduser(), compress,
                         on_progress=lambda c, i, n: st.update(f"[ink]Layer cache · {c} {i}/{n}"))
        ui.ok(f"{compress} layer cache ready in {ui.tilde(s.layer_cache)}")
    ui.ok("camera ready — /shoot")


# ---- settings ------------------------------------------------------------------------ #


def config_show() -> None:
    s = cfg.load()
    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style="accent")
    t.add_column(style="ink")
    for k, v in s.model_dump().items():
        t.add_row(k, str(v) if v != "" else "[dim]auto[/]")
    ui.console.print(t)
    ui.info(f"file: {cfg.config_path()}")


def config_set(key: str, value: str | None) -> cfg.Settings:
    s = cfg.load()
    if key not in cfg.Settings.model_fields:
        raise Abort(f"unknown setting '{key}'", "/config lists them")
    if value is None:
        ui.console.print(Text.assemble((f"  {key} ", "accent"), (str(getattr(s, key)) or "auto", "ink")))
        return s
    allowed = {"backend": BACKENDS, "quality": QUALITIES, "aspect": ASPECTS, "compress": ["none", "fp8"]}
    if key in allowed and value not in allowed[key]:
        raise Abort(f"{key} must be one of: {', '.join(allowed[key])}")
    data = s.model_dump()
    data[key] = "" if value in ("auto", "default") and key in ("text_model", "image_model") else value
    new = cfg.Settings(**data)
    new.save()
    ui.ok(f"{key} = {value}")
    return new


# ---- doctor -------------------------------------------------------------------------- #


def doctor() -> None:
    s = cfg.load()
    ui.section("Doctor", "⚕")
    rows: list[tuple[str, bool | None, str]] = []

    def check(name: str, good: bool | None, detail: str) -> None:
        rows.append((name, good, detail))

    check("ffmpeg", bool(shutil.which("ffmpeg")), shutil.which("ffmpeg") or "brew install ffmpeg")
    check("node / npx", bool(shutil.which("npx")), shutil.which("npx") or "brew install node")
    check("ChatGPT sign-in", codex_auth_file().exists(),
          str(codex_auth_file()) if codex_auth_file().exists() else "/login chatgpt")
    up = proxy_up(s.base_url)
    check("director proxy", up if up else None, s.base_url if up else "starts automatically on first use")
    if up:
        try:
            from .llm import LLM

            llm = LLM(s)
            llm.ensure()
            check("models", True, f"text {llm.text_model} · image {llm.image_model}")
        except Exception as e:  # noqa: BLE001
            check("models", False, str(e))

    try:
        import torch

        from .render.ltx import pick_device

        check("torch", True, f"{torch.__version__} · device {pick_device(s.device)}")
        import diffusers

        good = hasattr(diffusers, "LTX2ImageToVideoPipeline")
        check("diffusers LTX-2.5", good, diffusers.__version__ if good else "reinstall Auteur (needs diffusers main)")
    except ImportError:
        check("render engine", False, "installed without it — reinstall, or use /backend animatic")

    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        check("memory", True, f"{ram:.0f} GB — airstream keeps ~2 DiT blocks + small models resident")
    except (ValueError, OSError):
        pass
    from . import weights as W

    wdir = Path(s.weights).expanduser()
    good, _, detail = _weights_row(wdir)
    check("LTX-2.5 weights", good, detail)
    if W.present(wdir):
        try:
            from .render.ltx import verify

            gaps = {c: m for c, m in verify(wdir).items() if m}
            check("weight keys", not gaps, "every streamed tensor matched" if not gaps else
                  "; ".join(f"{c}: {len(m)} missing (e.g. {m[0]})" for c, m in gaps.items()))
        except Exception as e:  # noqa: BLE001
            check("weight keys", False, f"{type(e).__name__}: {e}")
    if s.compress != "none":
        from .render.ltx import STREAMED

        cache = Path(s.layer_cache).expanduser()
        have = all((cache / f"{c}-{s.compress}" / "manifest.json").exists() for c in STREAMED)
        check(f"{s.compress} layer cache", have or None, str(cache) if have else "built by /fetch")
    if s.backend == "ltx-pipelines":
        check("ltx-pipelines", bool(s.ltx2_repo), s.ltx2_repo or "/config ltx2_repo /path/to/LTX-2")
    check("camera", True, f"{s.backend} · {s.quality} · {s.aspect}")
    check("films", True, str(s.home_path))

    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(no_wrap=True)
    t.add_column(style="ink", no_wrap=True)
    t.add_column(style="dim")
    for name, good, detail in rows:
        mark = Text("  ✓", style="ok") if good else Text("  ✕", style="err") if good is False else Text("  ○", style="warn")
        t.add_row(mark, name, detail)
    ui.console.print(t)

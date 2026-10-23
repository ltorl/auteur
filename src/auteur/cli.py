"""auteur — run it bare for the interactive session, or script it with subcommands."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Annotated, Optional

import typer

from . import actions as A
from . import ui
from .actions import Abort
from .llm import ProxyError

app = typer.Typer(
    name="auteur",
    help="🎬 Auteur — an AI film director in your terminal.\n\n"
    "Run [bold]auteur[/] for the interactive session (type ideas and notes, /help for commands). "
    "The subcommands below do the same things non-interactively, for scripts.",
    rich_markup_mode="rich",
    add_completion=False,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
proxy_app = typer.Typer(help="Manage the openai-oauth director proxy.", no_args_is_help=True)
app.add_typer(proxy_app, name="proxy")

ProjectOpt = Annotated[Optional[str], typer.Option("--project", "-p", help="Film slug or folder (default: latest).")]
ShotsOpt = Annotated[Optional[str], typer.Option("--shots", "-s", help="Comma-separated shot ids, e.g. S01,S04.")]
ForceOpt = Annotated[bool, typer.Option("--force", "-f", help="Redo even if it already exists.")]
BackendOpt = Annotated[Optional[str], typer.Option("--backend", "-b", help="airstream | ltx-pipelines | animatic")]
QualityOpt = Annotated[Optional[str], typer.Option("--quality", "-q", help="draft | standard | hq (two-stage)")]


def guard(fn):
    """Known failures become one friendly line instead of a traceback."""

    @functools.wraps(fn)
    def run(*a, **kw):
        try:
            return fn(*a, **kw)
        except (Abort, ProxyError) as e:
            ui.err(str(e), e.hint)
            raise typer.Exit(1) from None
        except KeyboardInterrupt:
            ui.console.print()
            ui.warn("cut! (interrupted — progress so far is saved)")
            raise typer.Exit(130) from None
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            lines = str(e).splitlines() or [type(e).__name__]
            ui.err(lines[0])
            if len(lines) > 1:
                ui.console.print("\n".join(lines[1:]), style="dim")
            raise typer.Exit(1) from None

    return run


@app.callback()
def root(
    ctx: typer.Context,
    project: ProjectOpt = None,
    resume: Annotated[bool, typer.Option("--continue", "-c", help="Reopen your most recent film.")] = False,
    version: Annotated[bool, typer.Option("--version", "-v", help="Print the version.")] = False,
) -> None:
    ui.fit()
    if version:
        from . import __version__

        ui.console.print(f"auteur {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        from .repl import Session

        Session(project_ref=project, resume=resume).run()


# ---- film ---------------------------------------------------------------------------- #


@app.command()
@guard
def new(
    idea: Annotated[str, typer.Argument(help="The idea, in a sentence or a page.")],
    template: Annotated[str, typer.Option("--template", "-t", help="See `auteur templates`.")] = "short-film",
    seconds: Annotated[int, typer.Option("--duration", "-d", help="Target runtime in seconds.")] = 30,
    aspect: Annotated[Optional[str], typer.Option("--aspect", "-a", help=" | ".join(A.ASPECTS))] = None,
    style: Annotated[str, typer.Option("--style", help="Style notes, references, mood.")] = "",
) -> None:
    """✦ Write a treatment and shot list for a new film."""
    p = A.start_film(A.studio(), idea, template, seconds, aspect, style)
    A.next_steps(p)


@app.command()
@guard
def make(
    idea: Annotated[str, typer.Argument(help="The idea.")],
    template: Annotated[str, typer.Option("--template", "-t")] = "short-film",
    seconds: Annotated[int, typer.Option("--duration", "-d")] = 30,
    aspect: Annotated[Optional[str], typer.Option("--aspect", "-a")] = None,
    style: Annotated[str, typer.Option("--style")] = "",
    backend: BackendOpt = None,
    quality: QualityOpt = None,
    open_after: Annotated[bool, typer.Option("--open", "-o", help="Open the film when done.")] = False,
) -> None:
    """▶ Idea → finished film, no questions asked."""
    ui.banner()
    s = A.studio()
    p = A.start_film(s, idea, template, seconds, aspect, style)
    A.do_roll(s, p, backend, quality, open_after)


@app.command()
@guard
def pitch(project: ProjectOpt = None) -> None:
    """✦ Rewrite the treatment."""
    s = A.studio()
    A.do_pitch(s, A.find_project(s, project))


@app.command()
@guard
def plan(project: ProjectOpt = None) -> None:
    """▤ Re-plan the shot list."""
    s = A.studio()
    A.do_plan(s, A.find_project(s, project))


@app.command()
@guard
def cast(project: ProjectOpt = None, force: ForceOpt = False) -> None:
    """☺ Draw character reference sheets."""
    s = A.studio()
    A.do_cast(s, A.find_project(s, project), force)


@app.command()
@guard
def board(project: ProjectOpt = None, shots: ShotsOpt = None, force: ForceOpt = False) -> None:
    """▦ Storyboard keyframes, consistent with the cast sheets."""
    s = A.studio()
    A.do_board(s, A.find_project(s, project), A.split_ids(shots), force)


@app.command()
@guard
def shoot(project: ProjectOpt = None, shots: ShotsOpt = None, backend: BackendOpt = None,
          quality: QualityOpt = None, force: ForceOpt = False) -> None:
    """● Render shots with LTX-2.5 (or -b animatic for an instant preview)."""
    s = A.studio()
    A.do_shoot(s, A.find_project(s, project), A.split_ids(shots), backend, quality, force)


@app.command()
@guard
def cut(
    project: ProjectOpt = None,
    titles: Annotated[bool, typer.Option("--titles/--no-titles", help="Opening and end cards.")] = True,
    music: Annotated[Optional[Path], typer.Option("--music", "-m", help="Bed a music track under the film.")] = None,
    open_after: Annotated[bool, typer.Option("--open", "-o", help="Open the film when done.")] = False,
) -> None:
    """✂ Assemble the final film."""
    s = A.studio()
    A.do_cut(s, A.find_project(s, project), titles, music, open_after)


@app.command()
@guard
def revise(note: Annotated[str, typer.Argument(help='e.g. "make S03 a slow push-in and cut S05"')],
           project: ProjectOpt = None) -> None:
    """✎ Give the director a note; stale shots are flagged for redo."""
    s = A.studio()
    A.do_revise(s, A.find_project(s, project), note)


@app.command("shots")
@guard
def shots_cmd(project: ProjectOpt = None) -> None:
    """▤ Show the shot list."""
    s = A.studio()
    ui.console.print(ui.shot_table(A.find_project(s, project)))


@app.command()
@guard
def show(shot: Annotated[Optional[str], typer.Argument(help="A shot id for the close-up view.")] = None,
         project: ProjectOpt = None) -> None:
    """▦ Show the storyboard (or one shot in detail)."""
    s = A.studio()
    p = A.find_project(s, project)
    root = s.store.dir(p)
    if shot:
        ui.console.print(ui.shot_detail(p, p.shot(shot), root))
        return
    if p.treatment:
        ui.console.print(ui.treatment_panel(p.treatment, p))
    ui.console.print(ui.storyboard(p, root))


@app.command("list")
def list_cmd() -> None:
    """▶ List your films."""
    s = A.studio()
    films = s.store.all()
    if films:
        ui.console.print(ui.project_list(films))
    else:
        ui.info(f"no films yet in {s.store.home}")


@app.command()
def templates() -> None:
    """▤ Quick-start formats."""
    ui.console.print(A.templates_table())


# ---- setup ----------------------------------------------------------------------------- #


@app.command()
@guard
def login(account: Annotated[Optional[str], typer.Argument(help="chatgpt | hf")] = None) -> None:
    """🔑 Sign in: ChatGPT (the director) or Hugging Face (LTX-2.5 weights)."""
    A.login(account)


@app.command()
def status() -> None:
    """● Accounts, director and camera at a glance."""
    from . import config as cfg

    A.status(cfg.load())


@app.command()
@guard
def fetch(
    hq: Annotated[bool, typer.Option("--hq/--no-hq", help="Also fetch the ×2 latent upsampler.")] = True,
    compress: Annotated[Optional[str], typer.Option("--compress", "-c", help="Layer cache: fp8 | none")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """⬇ Download LTX-2.5 and build the AirLLM-style layer cache."""
    A.fetch(hq, compress, yes)


@app.command("config")
@guard
def config_cmd(key: Annotated[Optional[str], typer.Argument()] = None,
               value: Annotated[Optional[str], typer.Argument()] = None) -> None:
    """⚙ Show or set settings (e.g. `auteur config backend animatic`)."""
    if key is None:
        A.config_show()
    else:
        A.config_set(key, value)


@app.command()
def doctor() -> None:
    """⚕ Check everything Auteur needs."""
    A.doctor()


@proxy_app.command("start")
@guard
def proxy_start() -> None:
    """Start the proxy in the background."""
    A.proxy("start")


@proxy_app.command("stop")
@guard
def proxy_stop() -> None:
    """Stop the background proxy."""
    A.proxy("stop")


@proxy_app.command("status")
@guard
def proxy_status() -> None:
    """Is the director reachable?"""
    A.proxy("status")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

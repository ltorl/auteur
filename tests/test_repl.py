"""The interactive session, driven through prompt_toolkit's pipe input."""

import json

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from auteur.repl import COMMANDS, Session, SlashCompleter, opts, split

from .fake_proxy import serve


@pytest.fixture()
def env(tmp_path, monkeypatch):
    srv, url = serve()
    monkeypatch.setenv("AUTEUR_CONFIG", str(tmp_path / "cfg" / "config.json"))
    monkeypatch.setenv("AUTEUR_BASE_URL", url)
    monkeypatch.setenv("AUTEUR_HOME", str(tmp_path / "films"))
    monkeypatch.setenv("AUTEUR_AUTO_PROXY", "false")
    monkeypatch.setenv("COLUMNS", "120")
    yield srv, tmp_path
    srv.shutdown()


def drive(*lines, **kw) -> Session:
    """Type each line + Enter into the full-screen app; lines sent while busy are queued."""
    with create_pipe_input() as inp:
        inp.send_text("".join(line + "\r" for line in lines))
        s = Session(pt_input=inp, pt_output=DummyOutput(), **kw)
        s.run()
    return s


def test_full_film_from_the_prompt(env, capsys):
    srv, tmp = env
    s = drive(
        "/help",
        "a lamplighter's last night",          # free text with no film open = new film
        "/backend animatic",
        "/board",
        "lock off @S02 please",                 # free text with a film open = director note
        "/shoot",
        "/cut --no-titles",
        "/nonsense",
        "/exit",
    )
    out = s.transcript
    assert "Welcome to Auteur" in out and "/login" in out
    assert "THE LAST LAMPLIGHTER" in out
    assert "Locked off S02" in out
    assert "unknown command /nonsense" in out
    assert "that's a wrap" in out
    p = s.project
    assert p.final and all(x.status == "shot" for x in p.shots)
    saved = json.loads((tmp / "cfg" / "config.json").read_text())
    assert saved["backend"] == "animatic"           # /backend persisted
    assert (tmp / "cfg" / "history").exists()        # history lives next to the config


def test_errors_keep_the_session_alive(env, capsys):
    out = drive("/shots", "/shoot -q ultra", "/config nope 1", "/new -d", "/status", "/exit").transcript
    assert "no film yet" in out
    assert "unknown setting 'nope'" in out
    assert "-d needs a value" in out
    assert "ChatGPT" in out and "camera" in out


def test_resume_latest(env, capsys):
    drive("a lamplighter's last night", "/exit")
    s = drive("/shots", "/exit", resume=True)
    assert s.project and s.project.slug.startswith("a-lamplighter")
    assert "resuming" in s.transcript


def _complete(session, text):
    return [(c.text, c.display_meta_text) for c in SlashCompleter(session).get_completions(Document(text), None)]


def test_completion(env):
    s = drive("a lamplighter's last night", "/exit")
    names = [t for t, _ in _complete(s, "/")]
    assert names == ["/" + n for n in COMMANDS]
    assert ("/login", COMMANDS["login"].help) in _complete(s, "/lo")
    assert [t for t, _ in _complete(s, "/login ")] == ["chatgpt", "hf"]
    assert [t for t, _ in _complete(s, "/shoot -b a")] == ["airstream", "animatic"]
    assert "S01" in [t for t, _ in _complete(s, "/board ")]
    assert _complete(s, "make @S0")[0] == ("@S01", "Beat 1")


def test_parsing():
    assert split("the vault's heist") == ["the", "vault's", "heist"]
    assert opts(["S01", "-b", "animatic", "-f"], {"-b": "backend"}, {"-f": "force"}) == (
        ["S01"], {"backend": "animatic", "force": True})


def test_images_refused_still_makes_the_film(env, capsys, monkeypatch):
    """Real-world: ChatGPT plan without image generation answers 403 Forbidden."""
    srv, tmp = env
    srv.image_status = 403
    monkeypatch.setenv("AUTEUR_BACKEND", "animatic")
    s = drive("a lamplighter's last night", "/roll", "/board", "/exit")
    out = s.transcript
    assert out.count("ChatGPT refused image generation") == 2  # once for /roll, once for /board
    assert "Error code: 403" not in out                       # no per-shot error spam
    image_calls = [p for p, _ in srv.calls if "/images/" in p]
    assert 1 <= len(image_calls) <= 3                          # first wave only, then we stop asking
    assert s.project.final and all(x.status == "shot" and not x.keyframe for x in s.project.shots)


def test_roll_checks_the_camera_before_spending_calls(env, capsys, monkeypatch, tmp_path):
    srv, _ = env
    monkeypatch.setenv("AUTEUR_BACKEND", "airstream")
    monkeypatch.setenv("AUTEUR_WEIGHTS", str(tmp_path / "no-weights"))
    drive("a lamplighter's last night", "/exit")
    before = len(srv.calls)
    out = drive("/roll", "/exit").transcript
    assert "isn't downloaded yet" in out and "/fetch" in out
    assert "opened your latest film" in out                    # /roll with nothing open picks the latest
    assert len(srv.calls) == before                            # failed fast: no director or image calls


def test_typo_suggestions(env, capsys):
    out = drive("/rol", "/opem", "/storybord", "/exit").transcript
    assert "did you mean /roll?" in out
    assert "did you mean /open?" in out
    assert "did you mean /board?" in out                       # via the alias 'storyboard'


def drive_keys(*steps, **kw) -> Session:
    """Like drive(), but each step's keys are sent after a pause, so pickers
    raised by the previous command are on screen when the keys arrive."""
    import threading
    import time

    with create_pipe_input() as inp:
        def typist():
            for keys in steps:
                time.sleep(1.0)
                inp.send_text(keys)

        threading.Thread(target=typist, daemon=True).start()
        s = Session(pt_input=inp, pt_output=DummyOutput(), **kw)
        s.run()
    return s


DOWN = "\x1b[B"


def test_pickers_in_the_full_screen_app(env):
    """Regression: redirecting stdout into the log made every picker fall back
    to its non-interactive path (e.g. /open just printed the film list)."""
    srv, tmp = env
    drive("a lamplighter's last night", "/exit")
    s = drive_keys("/open\r", "\r", "/backend\r", DOWN + DOWN + "\r", "/quality\r", "\x1b", "/exit\r")
    out = s.transcript
    assert "? Open which film?" in out and "opened The Last Lamplighter" in out
    assert s.project and s.project.slug.startswith("a-lamplighter")
    assert "? Render with  animatic" in out
    assert json.loads((tmp / "cfg" / "config.json").read_text())["backend"] == "animatic"
    assert "? Quality  cancelled" in out  # Esc cancels without changing anything

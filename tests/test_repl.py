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
    out = capsys.readouterr().out
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
    drive("/shots", "/shoot -q ultra", "/config nope 1", "/new -d", "/status", "/exit")
    out = capsys.readouterr().out
    assert "no film open" in out
    assert "unknown setting 'nope'" in out
    assert "-d needs a value" in out
    assert "ChatGPT" in out and "camera" in out


def test_resume_latest(env, capsys):
    drive("a lamplighter's last night", "/exit")
    s = drive("/shots", "/exit", resume=True)
    assert s.project and s.project.slug.startswith("a-lamplighter")
    assert "resuming" in capsys.readouterr().out


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

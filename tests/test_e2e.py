"""Full production through the real CLI, against a fake openai-oauth proxy."""

import json
import os
import subprocess
import sys

import pytest

from auteur.render import media

from .fake_proxy import serve


@pytest.fixture()
def env(tmp_path):
    srv, url = serve()
    e = dict(os.environ)
    e.update(
        AUTEUR_CONFIG=str(tmp_path / "config.json"),
        AUTEUR_BASE_URL=url,
        AUTEUR_HOME=str(tmp_path / "films"),
        AUTEUR_BACKEND="animatic",
        AUTEUR_AUTO_PROXY="false",
        COLUMNS="120",
    )
    yield e, srv, tmp_path / "films"
    srv.shutdown()


def run(env, *args):
    proc = subprocess.run([sys.executable, "-m", "auteur", *args], env=env, capture_output=True, text=True,
                          timeout=300, stdin=subprocess.DEVNULL)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_make_film_end_to_end(env):
    e, srv, home = env
    out = run(e, "make", "an old lamplighter's final night", "-d", "10", "-a", "16:9")
    assert "THE LAST LAMPLIGHTER" in out
    proj = json.loads((home / "an-old-lamplighter-s-final-night" / "project.json").read_text())

    # director: best full-size model chosen, invalid JSON repaired on retry
    chats = [json.loads(b) for p, b in srv.calls if p.endswith("/chat/completions")]
    assert {c["model"] for c in chats} == {"gpt-5.6-sol"}
    assert sum("invalid" in c["messages"][-1]["content"] for c in chats) == 1

    # cast sheets per character (ids slugified), keyframes via edits when refs exist
    assert set(proj["cast_sheets"]) == {"elias", "nell"}
    assert sum(p.endswith("/images/edits") for p, _ in srv.calls) == 3
    assert all(s["status"] == "shot" and s["backend"] == "animatic" for s in proj["shots"])

    film = home / proj["slug"] / proj["final"]
    info = media.probe(film)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1920, 1080)
    assert media.has_audio(film)
    # 3 s title + 3 shots × 73 frames @24fps + 2.5 s end card − 3 × 0.5 s crossfades
    expected = 3 + 3 * 73 / 24 + 2.5 - 1.5
    assert abs(float(info["format"]["duration"]) - expected) < 0.3


def test_revise_invalidates_only_changed_shots(env):
    e, srv, home = env
    run(e, "make", "lamplighter", "-d", "10")
    run(e, "revise", "make S02 a locked-off shot")
    proj = json.loads((home / "lamplighter" / "project.json").read_text())
    by_id = {s["id"]: s for s in proj["shots"]}
    assert by_id["S02"]["camera"] == "static"
    assert by_id["S02"]["clip"] is None and by_id["S02"]["keyframe"]  # motion change keeps the frame
    assert by_id["S01"]["clip"] and by_id["S03"]["clip"]
    assert proj["final"] is None
    assert proj["chat"][-1]["text"].startswith("Locked off S02")

    run(e, "shoot")  # only S02 is re-rendered
    run(e, "cut", "--no-titles")
    proj = json.loads((home / "lamplighter" / "project.json").read_text())
    assert proj["final"]


def test_views_render(env):
    e, _, _ = env
    run(e, "make", "lamplighter", "-d", "10")
    assert "S03" in run(e, "shots")
    assert "TREATMENT" in run(e, "show")
    assert "LTX prompt" in run(e, "show", "S01")
    assert "lamplighter" in run(e, "list")
    assert "gpt-5.6-sol" in run(e, "doctor")

"""Render backend contract, LTX frame/resolution math."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

FPS = 24.0
MAX_FRAMES = 241  # ~10 s per shot keeps memory sane on laptops

# (draft, standard, hq). Everything is divisible by 64 so two-stage (half-res
# first pass) still lands on the VAE's 32-px grid.
RESOLUTIONS: dict[str, dict[str, tuple[int, int]]] = {
    "16:9": {"draft": (768, 448), "standard": (1024, 576), "hq": (1920, 1088)},
    "9:16": {"draft": (448, 768), "standard": (576, 1024), "hq": (1088, 1920)},
    "1:1": {"draft": (512, 512), "standard": (768, 768), "hq": (1280, 1280)},
    "4:3": {"draft": (640, 512), "standard": (1024, 768), "hq": (1536, 1152)},
    "2.39:1": {"draft": (896, 384), "standard": (1280, 576), "hq": (1920, 832)},
}
DELIVERY: dict[str, tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:3": (1440, 1080),
    "2.39:1": (1920, 804),
}
QUALITIES = ("draft", "standard", "hq")


def frames_for(seconds: float, fps: float = FPS) -> int:
    """LTX needs ``frames % 8 == 1``; round to the nearest valid count."""
    k = max(1, round(seconds * fps / 8))
    return min(8 * k + 1, MAX_FRAMES)


def seconds_for(frames: int, fps: float = FPS) -> float:
    return (frames - 1) / fps


def resolution(aspect: str, quality: str) -> tuple[int, int]:
    return RESOLUTIONS.get(aspect, RESOLUTIONS["16:9"])[quality]


@dataclass
class ShotJob:
    shot_id: str
    prompt: str
    out: Path
    seconds: float
    width: int
    height: int
    seed: int
    image: Path | None = None
    camera: str = ""  # used by the animatic backend
    fps: float = FPS

    @property
    def num_frames(self) -> int:
        return frames_for(self.seconds, self.fps)


@dataclass
class Event:
    kind: str  # phase | step | unit | done | log
    shot: str | None = None
    text: str = ""
    i: int = 0
    n: int = 0
    data: dict = field(default_factory=dict)


Emit = Callable[[Event], None]


class Backend(Protocol):
    name: str

    def render(self, jobs: list[ShotJob], emit: Emit) -> list[Path]: ...

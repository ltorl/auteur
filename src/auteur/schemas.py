"""The film's data model: treatment, cast, shot list, and project state."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Transition = Literal["cut", "dissolve", "fade", "whip"]
ShotStatus = Literal["planned", "boarded", "shot", "failed"]


def slugify(text: str, fallback: str = "film") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:48].strip("-") or fallback


# ---- what the director (LLM) writes --------------------------------------- #


class Character(BaseModel):
    id: str = Field(description="short slug, e.g. 'mara'")
    name: str
    role: str = ""
    look: str = Field(description="locked visual description: age, build, face, hair, wardrobe, props")
    voice: str = Field("", description="how they sound, for audio prompting")

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        return slugify(v, "character")


class Location(BaseModel):
    id: str
    name: str
    look: str

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        return slugify(v, "location")


class StyleBible(BaseModel):
    visual_style: str = Field(description="overall look, references, era")
    palette: str = ""
    lighting: str = ""
    camera_language: str = ""
    texture: str = Field("", description="film stock, grain, lens character")
    sound: str = Field("", description="ambience, music direction, mix")


class Treatment(BaseModel):
    title: str
    logline: str
    synopsis: str
    genre: str = ""
    tone: str = ""
    style: StyleBible
    characters: list[Character] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)


class Line(BaseModel):
    speaker: str
    line: str


class ShotPlan(BaseModel):
    id: str = Field(description="S01, S02, ...")
    scene: int = 1
    title: str
    duration: float = Field(description="seconds, 2-10")
    shot_size: str = Field(description="EWS / WS / MS / MCU / CU / ECU / OTS / POV / insert")
    camera: str = Field(description="movement, e.g. 'slow push in', 'handheld follow', 'static'")
    lens: str = ""
    characters: list[str] = Field(default_factory=list, description="character ids in frame")
    location: str = ""
    action: str
    dialogue: list[Line] = Field(default_factory=list)
    sound: str = ""
    keyframe_prompt: str = Field(description="the opening frame as a still image")
    motion_prompt: str = Field(description="LTX video prompt: one flowing paragraph")
    transition: Transition = "cut"

    @field_validator("duration")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return float(min(max(v, 1.5), 10.0))


class ShotList(BaseModel):
    shots: list[ShotPlan]


class Revision(BaseModel):
    reply: str = Field(description="what the director says back, 1-3 sentences")
    treatment: Treatment | None = None
    shots: list[ShotPlan] | None = None


# ---- project state --------------------------------------------------------- #


class Shot(ShotPlan):
    status: ShotStatus = "planned"
    seed: int = 0
    keyframe: str | None = None
    clip: str | None = None
    backend: str | None = None
    error: str | None = None


class Message(BaseModel):
    role: Literal["user", "director"]
    text: str
    at: datetime = Field(default_factory=datetime.now)


class Project(BaseModel):
    slug: str
    idea: str
    template: str = "short-film"
    aspect: str = "16:9"
    target_seconds: int = 30
    style_hint: str = ""
    created: datetime = Field(default_factory=datetime.now)
    updated: datetime = Field(default_factory=datetime.now)
    treatment: Treatment | None = None
    shots: list[Shot] = Field(default_factory=list)
    cast_sheets: dict[str, str] = Field(default_factory=dict)  # character id -> image path
    final: str | None = None
    chat: list[Message] = Field(default_factory=list)

    @property
    def title(self) -> str:
        return self.treatment.title if self.treatment else self.idea[:60]

    @property
    def runtime(self) -> float:
        return sum(s.duration for s in self.shots)

    def shot(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id.lower() == shot_id.lower():
                return s
        raise KeyError(shot_id)

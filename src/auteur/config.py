"""Settings: ~/.config/auteur/config.json, overridable with AUTEUR_* env vars."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from pydantic import BaseModel, Field


def config_path() -> Path:
    return Path(os.environ.get("AUTEUR_CONFIG", "~/.config/auteur/config.json")).expanduser()


def _default_home() -> str:
    movies = Path("~/Movies").expanduser()
    return str(movies / "Auteur") if platform.system() == "Darwin" and movies.exists() else "~/auteur"


class Settings(BaseModel):
    # director (openai-oauth proxy)
    base_url: str = "http://127.0.0.1:10531/v1"
    text_model: str = ""  # empty = best available from /v1/models
    image_model: str = ""
    reasoning_effort: str = ""  # low | medium | high, if the model takes it
    auto_proxy: bool = True  # start `npx openai-oauth` when it isn't running

    # studio
    home: str = Field(default_factory=_default_home)
    backend: str = "airstream"  # airstream | ltx-pipelines | animatic
    quality: str = "draft"  # draft | standard | hq
    aspect: str = "16:9"

    # airstream (LTX-2.5 diffusers weights, layer-streamed)
    weights: str = "~/.cache/auteur/LTX-2.5-Diffusers"
    layer_cache: str = "~/.cache/auteur/layers"
    compress: str = "fp8"  # none | fp8 — what `auteur fetch` builds and renders use
    device: str = "auto"
    text_batch: int = 4

    # ltx-pipelines (official CUDA path)
    ltx2_repo: str = ""
    ltx_checkpoints: str = ""

    @property
    def home_path(self) -> Path:
        return Path(self.home).expanduser()

    def save(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def load() -> Settings:
    data = {}
    path = config_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    for name, field in Settings.model_fields.items():
        env = os.environ.get(f"AUTEUR_{name.upper()}")
        if env is not None:
            data[name] = env.lower() in ("1", "true", "yes") if field.annotation is bool else env
    return Settings(**data)

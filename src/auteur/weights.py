"""Fetch exactly the LTX-2.5 diffusers files a render needs (and nothing twice).

The repo ships some components both sharded and unsharded (and two shardings
of the transformer); downloading by pattern would pull ~2x the bytes, so we
read each component's index and fetch only the files it references.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

REPO = "Lightricks/LTX-2.5-Diffusers"
CORE = ["transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder"]
HQ = ["latent_upsampler"]
SMALL = ["model_index.json", "scheduler/*", "tokenizer/*", "*/config.json", "*/generation_config.json",
         "*/*.index.json"]


class GatedError(RuntimeError):
    pass


def token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def plan(dest: Path, hq: bool = True) -> list[str]:
    """Download the small metadata files, then list the weight files to fetch."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        snapshot_download(REPO, local_dir=dest, allow_patterns=SMALL)
    except GatedRepoError as e:
        raise GatedError(str(e)) from e
    except HfHubHTTPError as e:
        if getattr(e.response, "status_code", None) in (401, 403):
            raise GatedError(str(e)) from e
        raise
    files: list[str] = []
    for comp in CORE + (HQ if hq else []):
        d = dest / comp
        idx = sorted(d.glob("*.safetensors.index.json"))
        if idx:
            files += sorted({f"{comp}/{v}" for v in json.loads(idx[0].read_text())["weight_map"].values()})
        elif comp in ("text_encoder",):
            files.append(f"{comp}/model.safetensors")
        else:
            files.append(f"{comp}/diffusion_pytorch_model.safetensors")
    return files


def sizes(files: list[str]) -> dict[str, int]:
    from huggingface_hub import HfApi

    out = {}
    for info in HfApi().get_paths_info(REPO, files):
        out[info.path] = getattr(info, "size", 0) or 0
    return out


def download(dest: Path, files: list[str], on_file: Callable[[str], None] | None = None) -> None:
    from huggingface_hub import hf_hub_download

    for f in files:
        if on_file:
            on_file(f)
        hf_hub_download(REPO, f, local_dir=dest)


def present(dest: Path) -> bool:
    return (dest / "model_index.json").exists() and all((dest / c).exists() for c in CORE)

"""Fetch exactly the LTX-2.5 diffusers files a render needs (and nothing twice).

The repo ships some components both sharded and unsharded (and two shardings
of the transformer); downloading by pattern would pull ~2x the bytes, so we
read each component's index and fetch only the files it references.

Downloads run in parallel, resume after interruption, and report one overall
progress figure (Hugging Face's own per-file bars are silenced). A marker file
is written only once every file is present, so a half-finished download is
never mistaken for a usable model.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = "Lightricks/LTX-2.5-Diffusers"
CORE = ["transformer", "text_encoder", "connectors", "vae", "audio_vae", "vocoder"]
HQ = ["latent_upsampler"]
SMALL = ["model_index.json", "scheduler/*", "tokenizer/*", "*/config.json", "*/generation_config.json",
         "*/*.index.json"]
PLAN_FILE = ".auteur-plan.json"
DONE_FILE = ".auteur-complete.json"


class GatedError(RuntimeError):
    pass


def quiet_hub() -> None:
    """Silence huggingface_hub / hf_xet progress bars; we draw our own."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:
        pass


def token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def plan(dest: Path, hq: bool = True) -> list[str]:
    """Download the small metadata files, then list the weight files to fetch."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    quiet_hub()
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
        elif comp == "text_encoder":
            files.append(f"{comp}/model.safetensors")
        else:
            files.append(f"{comp}/diffusion_pytorch_model.safetensors")
    return files


def sizes(files: list[str]) -> dict[str, int]:
    from huggingface_hub import HfApi

    return {info.path: getattr(info, "size", 0) or 0 for info in HfApi().get_paths_info(REPO, files)}


def save_plan(dest: Path, file_sizes: dict[str, int]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / PLAN_FILE).write_text(json.dumps(file_sizes))
    (dest / DONE_FILE).unlink(missing_ok=True)


@dataclass
class Progress:
    done_bytes: int
    total_bytes: int
    done_files: int
    total_files: int

    @property
    def fraction(self) -> float:
        return self.done_bytes / self.total_bytes if self.total_bytes else 0.0


def progress(dest: Path) -> Progress | None:
    """How much of the planned download is on disk (finished files + partials)."""
    f = dest / PLAN_FILE
    if not f.exists():
        return None
    planned: dict[str, int] = json.loads(f.read_text())
    done_bytes = done_files = 0
    for rel, size in planned.items():
        p = dest / rel
        if p.exists() and (not size or p.stat().st_size == size):
            done_bytes += size or p.stat().st_size
            done_files += 1
    partial_dir = dest / ".cache" / "huggingface" / "download"
    if partial_dir.exists():
        for part in partial_dir.rglob("*.incomplete"):
            try:
                done_bytes += part.stat().st_size
            except OSError:
                pass
    total = sum(planned.values())
    return Progress(min(done_bytes, total), total, done_files, len(planned))


def download(dest: Path, files: list[str], on_progress: Callable[[Progress], None] | None = None,
             workers: int = 4) -> None:
    """Parallel, resumable download of ``files``; marks the folder complete at the end."""
    from huggingface_hub import snapshot_download

    quiet_hub()
    stop = threading.Event()

    def poll():
        while not stop.wait(0.5):
            p = progress(dest)
            if p and on_progress:
                on_progress(p)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    try:
        snapshot_download(REPO, local_dir=dest, allow_patterns=files, max_workers=workers)
    finally:
        stop.set()
        watcher.join(timeout=2)
    p = progress(dest)
    if on_progress and p:
        on_progress(p)
    missing = [f for f in files if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} files did not download (e.g. {missing[0]}) — run /fetch again")
    (dest / DONE_FILE).write_text(json.dumps({"files": files, "at": time.time()}))


def present(dest: Path) -> bool:
    """True only for a complete download (marker written after the last file)."""
    marker = dest / DONE_FILE
    if not marker.exists():
        return False
    files = json.loads(marker.read_text()).get("files", [])
    return all((dest / f).exists() for f in files)

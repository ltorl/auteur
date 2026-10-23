"""Official Lightricks ``ltx-pipelines`` backend (CUDA boxes).

Runs ``python -m ltx_pipelines.distilled`` from a checkout of
github.com/Lightricks/LTX-2 against the per-component ``Lightricks/LTX-2.5``
checkpoints, with ``--offload disk`` (LTX's own disk streaming) and
``--quantization fp8-cast`` for low-VRAM GPUs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .base import Emit, Event, ShotJob

FILES = {
    "--transformer-path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "--text-encoder-path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "--video-vae-path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "--audio-vae-path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "--spatial-upsampler-path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}
STEP_RE = re.compile(r"(\d+)/(\d+)")


class LTXOfficialBackend:
    name = "ltx-pipelines"

    def __init__(self, repo: str | Path, checkpoints: str | Path, offload: str = "disk",
                 quantization: str | None = "fp8-cast"):
        self.repo = Path(repo).expanduser()
        self.ckpt = Path(checkpoints).expanduser()
        self.offload = offload
        self.quantization = quantization
        if not (self.repo / "packages" / "ltx-pipelines").exists():
            raise FileNotFoundError(f"{self.repo} is not a Lightricks/LTX-2 checkout")
        missing = [p for p in FILES.values() if not (self.ckpt / p).exists()]
        if missing:
            raise FileNotFoundError(f"missing LTX-2.5 checkpoints in {self.ckpt}: {', '.join(missing)}")

    def command(self, job: ShotJob) -> list[str]:
        uv = shutil.which("uv") or "uv"
        cmd = [uv, "run", "--directory", str(self.repo), "python", "-m", "ltx_pipelines.distilled"]
        for flag, rel in FILES.items():
            cmd += [flag, str(self.ckpt / rel)]
        cmd += [
            "--prompt", job.prompt,
            "--seed", str(job.seed),
            "--num-frames", str(job.num_frames),
            "--width", str(job.width),
            "--height", str(job.height),
            "--frame-rate", str(job.fps),
            "--offload", self.offload,
            "--output-path", str(job.out),
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if job.image and Path(job.image).exists():
            cmd += ["--image", str(job.image), "0", "1.0"]
        return cmd

    def render(self, jobs: list[ShotJob], emit: Emit) -> list[Path]:
        done = []
        for job in jobs:
            emit(Event("phase", shot=job.shot_id, text=f"ltx-pipelines · {job.shot_id}"))
            job.out.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(self.command(job), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            tail: list[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                tail = (tail + [line])[-20:]
                m = STEP_RE.search(line)
                if m and "it" in line:
                    emit(Event("step", shot=job.shot_id, i=int(m.group(1)), n=int(m.group(2))))
                elif line:
                    emit(Event("log", shot=job.shot_id, text=line[-160:]))
            if proc.wait() != 0 or not job.out.exists():
                raise RuntimeError("ltx-pipelines failed:\n" + "\n".join(tail))
            done.append(job.out)
            emit(Event("done", shot=job.shot_id, text=str(job.out)))
        return done

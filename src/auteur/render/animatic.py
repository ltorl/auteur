"""Animatic backend: camera-aware Ken Burns moves over storyboard keyframes.

Renders in seconds with nothing but ffmpeg — for previewing pacing before
spending GPU time, and as a fallback when no LTX weights are around.
"""

from __future__ import annotations

from pathlib import Path

from .base import Emit, Event, ShotJob
from .media import AUDIO_RATE, run_ffmpeg

# camera keyword -> (zoom_start, zoom_end, x_drift, y_drift). Drift is where the
# crop window travels, in [-1, 1]: a pan left moves the window left (-1).
MOVES: list[tuple[tuple[str, ...], tuple[float, float, float, float]]] = [
    (("pull back", "pull-back", "dolly out", "zoom out", "reveal", "crane up"), (1.25, 1.0, 0, 0)),
    (("push in", "push-in", "dolly in", "zoom in", "creep"), (1.0, 1.25, 0, 0)),
    (("pan left", "truck left", "track left"), (1.15, 1.15, -1, 0)),
    (("pan right", "truck right", "track right", "tracking"), (1.15, 1.15, 1, 0)),
    (("tilt up", "pedestal up"), (1.15, 1.15, 0, -1)),
    (("tilt down", "pedestal down", "crane down"), (1.15, 1.15, 0, 1)),
    (("orbit", "arc"), (1.12, 1.18, -1, 0)),
    (("handheld", "shaky"), (1.1, 1.14, 0.3, 0.2)),
    (("static", "locked", "tripod"), (1.04, 1.06, 0, 0)),
]


def move_for(camera: str) -> tuple[float, float, float, float]:
    c = camera.lower()
    for keys, move in MOVES:
        if any(k in c for k in keys):
            return move
    return (1.0, 1.12, 0, 0)


def placeholder(job: ShotJob, path: Path) -> Path:
    from ..imagery import slate_card

    slate_card(job.shot_id, job.prompt, job.width, job.height).save(path)
    return path


class AnimaticBackend:
    name = "animatic"

    def render(self, jobs: list[ShotJob], emit: Emit) -> list[Path]:
        done = []
        for job in jobs:
            emit(Event("phase", shot=job.shot_id, text=f"Animatic {job.shot_id}"))
            src = Path(job.image) if job.image and Path(job.image).exists() else None
            if src is None:
                src = placeholder(job, job.out.with_suffix(".slate.png"))
            done.append(self.render_one(job, src))
            emit(Event("done", shot=job.shot_id, text=str(job.out)))
        return done

    def render_one(self, job: ShotJob, image: Path) -> Path:
        w, h = job.width, job.height
        frames = job.num_frames
        z0, z1, dx, dy = move_for(job.camera)
        # Upscale first so zoompan's integer crop offsets don't jitter.
        big = (w * 4, h * 4)
        n = max(frames - 1, 1)
        zoom = f"{z0}+({z1}-{z0})*on/{n}"
        x = f"(iw-iw/zoom)/2+({dx})*(iw-iw/zoom)/2*(2*on/{n}-1)"
        y = f"(ih-ih/zoom)/2+({dy})*(ih-ih/zoom)/2*(2*on/{n}-1)"
        vf = (
            f"scale={big[0]}:{big[1]}:force_original_aspect_ratio=increase,crop={big[0]}:{big[1]},"
            f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={job.fps},"
            f"format=yuv420p"
        )
        seconds = frames / job.fps
        job.out.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(
            [
                "-i", str(image),
                "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo",
                "-vf", vf, "-frames:v", str(frames),
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-c:a", "aac", "-ar", str(AUDIO_RATE), "-t", f"{seconds:.3f}", str(job.out),
            ]
        )
        return job.out

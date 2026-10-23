"""ffmpeg helpers: encode frames+audio, probe, normalise, assemble a cut."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUDIO_RATE = 48000


class FFmpegError(RuntimeError):
    pass


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegError("ffmpeg not found — install it (e.g. `brew install ffmpeg`)")
    return exe


def run_ffmpeg(args: Sequence[str], stdin: bytes | None = None) -> None:
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)]
    proc = subprocess.run(cmd, input=stdin, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-12:]
        raise FFmpegError("ffmpeg failed:\n" + "\n".join(tail))


def probe(path: str | Path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise FFmpegError("ffprobe not found")
    out = subprocess.run(
        [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        check=True,
    ).stdout
    return json.loads(out)


def duration(path: str | Path) -> float:
    return float(probe(path)["format"]["duration"])


def has_audio(path: str | Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe(path)["streams"])


def _channels_first(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    if a.shape[0] > 8 and a.shape[-1] <= 8:  # (samples, channels)
        a = a.T
    return a


def fit_audio(audio: np.ndarray, samples: int) -> np.ndarray:
    """Pad with silence / trim so the track spans exactly the picture."""
    a = _channels_first(audio)
    if a.shape[1] >= samples:
        return a[:, :samples]
    return np.pad(a, ((0, 0), (0, samples - a.shape[1])))


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    """``audio``: float array shaped (channels, samples) or (samples,)."""
    a = _channels_first(audio)
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def encode_video(
    frames: np.ndarray,
    out: str | Path,
    fps: float,
    audio: np.ndarray | None = None,
    audio_rate: int | None = None,
    crf: int = 17,
) -> Path:
    """``frames``: (F, H, W, 3) uint8 or float in [0, 1]."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = np.asarray(frames)
    if f.dtype != np.uint8:
        f = (np.clip(f, 0, 1) * 255).round().astype(np.uint8)
    n, h, w, _ = f.shape
    args = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
    with tempfile.TemporaryDirectory() as tmp:
        if audio is not None and audio_rate:
            wav = Path(tmp) / "a.wav"
            write_wav(wav, fit_audio(audio, round(n / fps * audio_rate)), audio_rate)
            args += ["-i", str(wav), "-c:a", "aac", "-b:a", "192k"]
        args += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]
        args += ["-movflags", "+faststart", str(out)]
        run_ffmpeg(args, stdin=np.ascontiguousarray(f).tobytes())
    return out


def normalize_clip(src: Path, dst: Path, width: int, height: int, fps: float) -> Path:
    """Scale/pad to the delivery format, constant fps, and always carry a stereo track."""
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},format=yuv420p"
    )
    args = ["-i", str(src)]
    if has_audio(src):
        amap = ["-map", "0:v:0", "-map", "0:a:0"]
    else:
        args += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
        amap = ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    args += ["-vf", vf, *amap, "-c:v", "libx264", "-crf", "16", "-preset", "medium"]
    args += ["-c:a", "aac", "-ar", str(AUDIO_RATE), "-ac", "2", "-b:a", "192k", str(dst)]
    run_ffmpeg(args)
    return dst


def still_clip(image: Path, dst: Path, seconds: float, width: int, height: int, fps: float) -> Path:
    """A still image held for ``seconds`` with a silent track (title cards)."""
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},format=yuv420p"
    )
    run_ffmpeg(
        [
            "-loop", "1", "-t", f"{seconds}", "-i", str(image),
            "-f", "lavfi", "-t", f"{seconds}", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo",
            "-vf", vf, "-c:v", "libx264", "-crf", "16", "-c:a", "aac", "-ar", str(AUDIO_RATE),
            "-shortest", str(dst),
        ]
    )
    return dst


@dataclass
class Segment:
    path: Path
    transition: str = "cut"  # how this segment enters: cut | dissolve | fade | whip


XFADE = {"dissolve": "fade", "fade": "fadeblack", "whip": "slideleft"}


def assemble(
    segments: Sequence[Segment],
    out: Path,
    transition_seconds: float = 0.5,
    fade_ends: bool = True,
    music: Path | None = None,
    music_gain_db: float = -16.0,
) -> Path:
    """Join normalised segments with cuts / crossfades into one film."""
    if not segments:
        raise ValueError("nothing to assemble")
    durations = [duration(s.path) for s in segments]
    inputs: list[str] = []
    for s in segments:
        inputs += ["-i", str(s.path)]

    # concat emits the AV timebase and xfade insists both inputs match, so
    # put every input on it up front.
    parts: list[str] = []
    for i in range(len(segments)):
        parts.append(f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS[in{i}]")
        parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[ia{i}]")
    v, a, total = "[in0]", "[ia0]", durations[0]
    for i, seg in enumerate(segments[1:], start=1):
        kind = XFADE.get(seg.transition)
        t = min(transition_seconds, durations[i] / 2, total / 2) if kind else 0.0
        nv, na = f"[v{i}]", f"[a{i}]"
        if kind and t > 0.05:
            parts.append(f"{v}[in{i}]xfade=transition={kind}:duration={t:.3f}:offset={total - t:.3f}{nv}")
            parts.append(f"{a}[ia{i}]acrossfade=d={t:.3f}{na}")
            total += durations[i] - t
        else:
            parts.append(f"{v}{a}[in{i}][ia{i}]concat=n=2:v=1:a=1{nv}{na}")
            total += durations[i]
        v, a = nv, na

    if fade_ends:
        parts.append(f"{v}fade=t=in:st=0:d=0.6,fade=t=out:st={max(total - 0.8, 0):.3f}:d=0.8[vout]")
        parts.append(f"{a}afade=t=in:st=0:d=0.4,afade=t=out:st={max(total - 0.8, 0):.3f}:d=0.8[aout]")
        v, a = "[vout]", "[aout]"

    if music is not None:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        m = len(segments)
        parts.append(
            f"[{m}:a]volume={music_gain_db}dB,atrim=0:{total:.3f},"
            f"afade=t=out:st={max(total - 1.5, 0):.3f}:d=1.5[mus]"
        )
        parts.append(f"{a}[mus]amix=inputs=2:duration=first:normalize=0[amix]")
        a = "[amix]"

    args = [*inputs]
    if parts:
        args += ["-filter_complex", ";".join(parts), "-map", v, "-map", a]
    else:
        args += ["-map", "0:v", "-map", "0:a"]
    args += ["-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    run_ffmpeg(args)
    return out

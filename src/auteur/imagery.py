"""Stills: character sheets and storyboard keyframes (gpt-image via openai-oauth)."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageOps


def fit_to(img: Image.Image, width: int, height: int) -> Image.Image:
    """Centre-crop to the target aspect, then resize."""
    return ImageOps.fit(img, (width, height), Image.LANCZOS, centering=(0.5, 0.5))


def api_size(aspect: str) -> str:
    """Closest size the image models accept for a film aspect ratio."""
    return {"9:16": "1024x1536", "1:1": "1024x1024"}.get(aspect, "1536x1024")


def _save_b64(b64: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(b64))
    # normalise whatever came back (webp/jpeg/png) into a real PNG
    Image.open(out).convert("RGB").save(out, format="PNG")
    return out


def generate(client, model: str, prompt: str, size: str, out: Path) -> Path:
    resp = client.images.generate(model=model, prompt=prompt, size=size, n=1)
    return _save_b64(resp.data[0].b64_json, out)


def edit(client, model: str, prompt: str, refs: Sequence[Path], size: str, out: Path) -> Path:
    """Generate with reference images (character consistency)."""
    files = [open(p, "rb") for p in refs]
    try:
        resp = client.images.edit(model=model, image=files if len(files) > 1 else files[0],
                                  prompt=prompt, size=size, n=1)
    finally:
        for f in files:
            f.close()
    return _save_b64(resp.data[0].b64_json, out)


# ---- cards (drawn locally) ------------------------------------------------- #

SERIF = [
    "/System/Library/Fonts/Supplemental/Didot.ttc",
    "/System/Library/Fonts/NewYork.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif.ttf",
]
SANS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


def font(candidates: list[str], size: int):
    from PIL import ImageFont

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _wrap(draw, text: str | list[str], fnt, width: int, sep: str = " ") -> list[str]:
    lines, cur = [], ""
    for word in text.split() if isinstance(text, str) else text:
        trial = f"{cur}{sep}{word}" if cur else word
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def title_card(title: str, subtitle: str, width: int, height: int) -> Image.Image:
    from PIL import ImageDraw

    img = Image.new("RGB", (width, height), (8, 8, 10))
    d = ImageDraw.Draw(img)
    big = font(SERIF, max(28, height // 9))
    small = font(SANS, max(14, height // 40))
    words = [" ".join(w) for w in title.upper().split()]  # letter-spaced, wrapped between words
    lines = _wrap(d, words, big, int(width * 0.85), sep="   ") or [title.upper()]
    lh = big.size * 1.25
    y = height / 2 - lh * len(lines) / 2 - small.size
    for line in lines:
        d.text((width / 2, y), line, font=big, fill=(236, 230, 218), anchor="ma")
        y += lh
    rule_w = width * 0.08
    d.line([(width / 2 - rule_w, y + lh * 0.25), (width / 2 + rule_w, y + lh * 0.25)], fill=(255, 176, 0), width=2)
    for line in _wrap(d, subtitle, small, int(width * 0.7))[:3]:
        y += small.size * 1.6
        d.text((width / 2, y + lh * 0.3), line, font=small, fill=(150, 146, 140), anchor="ma")
    return img


def slate_card(shot_id: str, text: str, width: int, height: int) -> Image.Image:
    """A clapperboard placeholder frame for shots with no keyframe yet."""
    from PIL import ImageDraw

    img = Image.new("RGB", (width, height), (24, 24, 28))
    d = ImageDraw.Draw(img)
    band = height // 7
    stripe = band
    for i, x in enumerate(range(-band, width + band, stripe)):
        color = (236, 230, 218) if i % 2 == 0 else (24, 24, 28)
        d.polygon([(x, 0), (x + stripe, 0), (x + stripe - band // 2, band), (x - band // 2, band)], fill=color)
    head = font(SANS, max(16, height // 10))
    body = font(SANS, max(12, height // 28))
    d.text((width * 0.06, band * 1.4), shot_id, font=head, fill=(255, 176, 0))
    y = band * 1.4 + head.size * 1.4
    for line in _wrap(d, text, body, int(width * 0.88))[: max(1, int((height - y) / (body.size * 1.4)) - 1)]:
        d.text((width * 0.06, y), line, font=body, fill=(200, 196, 188))
        y += body.size * 1.4
    return img

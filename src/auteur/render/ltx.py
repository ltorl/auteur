"""LTX-2.5 via diffusers, with the heavy components layer-streamed from disk.

Weights are the diffusers layout of ``Lightricks/LTX-2.5-Diffusers``. The 22B
DiT, the Gemma 4 text encoder and the 6 GB text connectors are never loaded
whole: they're built on ``meta`` and fed one block at a time from disk by
:class:`~auteur.render.airstream.AirStream`. The small ones (VAEs, vocoder,
upsampler) load normally.

Shots are rendered in two passes so the text encoder streams once per batch of
prompts rather than once per shot.
"""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path

import numpy as np

from .airstream import AirStream, build_layer_cache, open_source
from .base import Emit, Event, ShotJob
from .media import encode_video

STREAMED = ("transformer", "text_encoder", "connectors")  # 38 + 24 + 6 GB
SKIPPED = ("prompt_enhancer", "processor", "duration_head")


def pick_device(pref: str | None = None) -> str:
    import torch

    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_class(library: str, name: str):
    for mod in (library, f"diffusers.pipelines.{library}", "diffusers", "transformers"):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        if hasattr(m, name):
            return getattr(m, name)
    raise ImportError(f"cannot import {library}.{name} — is diffusers installed from main?")


def model_index(weights: Path) -> dict:
    return json.loads((weights / "model_index.json").read_text())


def build_meta(weights: Path, component: str):
    """Instantiate a component on ``meta`` from its config — zero memory."""
    from accelerate import init_empty_weights

    lib, cls_name = model_index(weights)[component]
    cls = _resolve_class(lib, cls_name)
    folder = weights / component
    with init_empty_weights():
        if lib == "transformers":
            from transformers import AutoConfig

            model = cls._from_config(AutoConfig.from_pretrained(folder))
        else:
            model = cls.from_config(cls.load_config(folder))
    return model.eval()


def _streamed_pipeline_class(base):
    """The pipeline asks its modules for an execution device; ours live on meta."""

    class Streamed(base):
        @property
        def _execution_device(self):
            import torch

            return torch.device(self._auteur_device)

    Streamed.__name__ = base.__name__
    return Streamed


def cache_dir_for(cache_root: Path, component: str, compress: str) -> Path:
    return cache_root / f"{component}-{compress}"


def verify(weights: Path) -> dict[str, list[str]]:
    """Check every streamed tensor can be found in the checkpoint, without loading it."""
    from .airstream import SafetensorsSource, missing_weights

    return {c: missing_weights(build_meta(weights, c), SafetensorsSource(weights / c)) for c in STREAMED}


def build_caches(weights: Path, cache_root: Path, compress: str, on_progress=None) -> list[Path]:
    """Pre-split streamed components into per-block files (optionally fp8)."""
    out = []
    for comp in STREAMED:
        target = cache_dir_for(cache_root, comp, compress)
        if (target / "manifest.json").exists():
            out.append(target)
            continue
        model = build_meta(weights, comp)
        cb = (lambda i, n, c=comp: on_progress(c, i, n)) if on_progress else None
        out.append(build_layer_cache(model, weights / comp, target, compress=compress, on_progress=cb))
    return out


class LTXStreamEngine:
    name = "airstream"

    def __init__(
        self,
        weights: str | Path,
        device: str | None = None,
        cache_root: str | Path | None = None,
        compress: str = "none",
        two_stage: bool = False,
        prefetch: bool = True,
        text_batch: int = 4,
    ):
        import torch

        self.weights = Path(weights).expanduser()
        if not (self.weights / "model_index.json").exists():
            raise FileNotFoundError(
                f"LTX-2.5 weights not found in {self.weights} — run `auteur fetch` first"
            )
        self.device = pick_device(device)
        self.dtype = torch.bfloat16
        self.cache_root = Path(cache_root).expanduser() if cache_root else None
        self.compress = compress
        self.two_stage = two_stage
        self.prefetch = prefetch
        self.text_batch = max(1, text_batch)
        self.pipe = None
        self._streams: dict[str, AirStream] = {}
        self._emit: Emit = lambda e: None

    # -- setup -------------------------------------------------------------- #
    def _source(self, component: str):
        if self.cache_root:
            cached = cache_dir_for(self.cache_root, component, self.compress)
            if (cached / "manifest.json").exists():
                return open_source(cached)
        return open_source(self.weights / component)

    def _stream(self, component: str, model) -> AirStream:
        label = {"transformer": "DiT", "text_encoder": "text encoder", "connectors": "connectors"}[component]

        def on_unit(i, n, _name):
            self._emit(Event("unit", i=i + 1, n=n, text=label))

        s = AirStream(model, self._source(component), self.device, self.dtype, self.prefetch, on_unit)
        self._streams[component] = s
        return s

    def load(self) -> None:
        if self.pipe is not None:
            return
        import torch

        self._emit(Event("phase", text="Assembling LTX-2.5 (layer-streamed)"))
        from diffusers import ModelMixin
        from transformers import PreTrainedModel

        comps = {}
        for name, spec in model_index(self.weights).items():
            if name.startswith("_") or not isinstance(spec, list) or spec[0] is None:
                continue
            if name in SKIPPED:
                comps[name] = None
            elif name in STREAMED:
                comps[name] = build_meta(self.weights, name)
            else:
                lib, cls_name = spec
                cls = _resolve_class(lib, cls_name)
                kwargs = {"dtype": self.dtype} if issubclass(cls, (ModelMixin, PreTrainedModel)) else {}
                if lib == "transformers":
                    mod = cls.from_pretrained(self.weights / name, **kwargs)
                else:
                    mod = cls.from_pretrained(self.weights, subfolder=name, **kwargs)
                if isinstance(mod, torch.nn.Module):
                    mod = mod.to(self.device).eval()
                comps[name] = mod
        self._comps = comps
        self.pipe = self._make_pipeline("LTX2ImageToVideoPipeline")
        self.pipe.vae.enable_tiling()
        self._stream("transformer", self.pipe.transformer)
        self._stream("connectors", self.pipe.connectors)

    def _make_pipeline(self, class_name: str):
        import diffusers

        cls = _streamed_pipeline_class(getattr(diffusers, class_name))
        accepted = set(cls.__init__.__code__.co_varnames)
        pipe = cls(**{k: v for k, v in self._comps.items() if k in accepted})
        pipe._auteur_device = self.device
        pipe.set_progress_bar_config(disable=True)
        return pipe

    @property
    def t2v(self):
        """Text-to-video twin sharing every module, for shots without a keyframe."""
        if getattr(self, "_t2v", None) is None:
            self._comps["transformer"] = self.pipe.transformer
            self._t2v = self._make_pipeline("LTX2Pipeline")
        return self._t2v

    # -- text --------------------------------------------------------------- #
    def encode(self, prompts: list[str]):
        """Stream the text encoder once per batch; returns [(embeds, mask), ...]."""
        import torch

        self.load()
        te = build_meta(self.weights, "text_encoder")
        self.pipe.text_encoder = te
        stream = self._stream("text_encoder", te)
        out = []
        try:
            for start in range(0, len(prompts), self.text_batch):
                chunk = prompts[start : start + self.text_batch]
                self._emit(Event("phase", text=f"Reading the script ({start + len(chunk)}/{len(prompts)} prompts)"))
                with torch.inference_mode():
                    embeds, mask, _, _ = self.pipe.encode_prompt(
                        prompt=chunk,
                        do_classifier_free_guidance=False,
                        device=torch.device(self.device),
                    )
                out += [(embeds[i : i + 1].cpu(), mask[i : i + 1].cpu()) for i in range(len(chunk))]
        finally:
            stream.release()
            self._streams.pop("text_encoder", None)
            gc.collect()
        return out

    # -- video -------------------------------------------------------------- #
    def _call(self, job: ShotJob, embeds, image, width, height, generator, **kw):
        import torch
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        e, m = embeds
        kw.setdefault("sigmas", DISTILLED_SIGMA_VALUES)
        total = len(kw["sigmas"])

        def on_step(pipe, step, t, tensors):
            self._emit(Event("step", shot=job.shot_id, i=step + 1, n=total))
            return {}

        if image is not None:
            kw["image"] = image
        pipe = self.pipe if image is not None else self.t2v
        with torch.inference_mode():
            return pipe(
                prompt_embeds=e.to(self.device),
                prompt_attention_mask=m.to(self.device),
                width=width,
                height=height,
                num_frames=job.num_frames,
                frame_rate=job.fps,
                guidance_scale=1.0,
                audio_guidance_scale=1.0,
                stg_scale=0.0,
                audio_stg_scale=0.0,
                modality_scale=1.0,
                audio_modality_scale=1.0,
                generator=generator,
                callback_on_step_end=on_step,
                return_dict=False,
                **kw,
            )

    def _upsampler(self):
        if getattr(self, "_up", None) is None:
            import torch
            from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

            up = LTX2LatentUpsamplerModel.from_pretrained(
                self.weights, subfolder="latent_upsampler", dtype=torch.bfloat16
            ).to(self.device)
            self._up = LTX2LatentUpsamplePipeline(vae=self.pipe.vae, latent_upsampler=up)
            self._up.set_progress_bar_config(disable=True)
        return self._up

    def render_one(self, job: ShotJob, embeds) -> Path:
        import torch
        from PIL import Image

        from ..imagery import fit_to

        image = None
        if job.image and Path(job.image).exists():
            image = fit_to(Image.open(job.image).convert("RGB"), job.width, job.height)
        generator = torch.Generator("cpu").manual_seed(job.seed)

        if self.two_stage:
            from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

            self._emit(Event("phase", shot=job.shot_id, text="Stage 1 · half resolution"))
            half = image.resize((job.width // 2, job.height // 2), Image.LANCZOS) if image else None
            video_latent, audio_latent = self._call(
                job, embeds, half, job.width // 2, job.height // 2, generator, output_type="latent"
            )
            self._emit(Event("phase", shot=job.shot_id, text="Upscaling latents ×2"))
            with torch.inference_mode():
                video_latent = self._upsampler()(
                    latents=video_latent, latents_normalized=False, output_type="latent", return_dict=False
                )[0]
            self._emit(Event("phase", shot=job.shot_id, text="Stage 2 · refine at full resolution"))
            video, audio = self._call(
                job,
                embeds,
                image,
                job.width,
                job.height,
                generator,
                latents=video_latent,
                audio_latents=audio_latent,
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                output_type="np",
            )
        else:
            video, audio = self._call(job, embeds, image, job.width, job.height, generator, output_type="np")

        self._emit(Event("phase", shot=job.shot_id, text="Encoding"))
        frames = np.asarray(video[0])
        track = None
        if audio is not None and len(audio):
            a = audio[0]
            track = a.float().cpu().numpy() if hasattr(a, "float") else np.asarray(a, dtype=np.float32)
        rate = int(getattr(self.pipe.vocoder.config, "output_sampling_rate", 24000))
        return encode_video(frames, job.out, job.fps, track, rate)

    def render(self, jobs: list[ShotJob], emit: Emit) -> list[Path]:
        self._emit = emit
        self.load()
        embeds = self.encode([j.prompt for j in jobs])
        done = []
        for job, e in zip(jobs, embeds):
            emit(Event("phase", shot=job.shot_id, text=f"Shooting {job.shot_id}"))
            done.append(self.render_one(job, e))
            emit(Event("done", shot=job.shot_id, text=str(job.out)))
        return done

    def stats(self) -> dict:
        s = self._streams.get("transformer")
        return {} if s is None else {"GB read": s.stats.bytes_read / 1e9, "GB/s": s.stats.gbps}

    def close(self) -> None:
        for s in list(self._streams.values()):
            s.release()
        self._streams.clear()
        self.pipe = self._t2v = self._up = None
        gc.collect()

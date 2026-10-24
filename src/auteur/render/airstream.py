"""AirStream — AirLLM-style layer streaming for PyTorch models.

A model is built on the ``meta`` device (no memory), then each repeated block
(transformer layer) is materialised from disk right before its forward pass and
dropped again right after. Only one or two blocks are ever resident, so a 22B
DiT plus a 12B text encoder run on a 16 GB laptop at the cost of disk I/O.

Extras over a naive streamer:

* background prefetch of the next block, learned from the observed call order;
* an optional per-block *layer cache* (one contiguous file per block), with
  per-row ``fp8`` compression that halves the bytes read every denoising step;
* checkpoint/model key reconciliation (prefix differences, tied weights).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

FP8 = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0
SCALE_SUFFIX = "::scale"
CACHE_FORMAT = 1


# --------------------------------------------------------------------------- #
# Weight sources
# --------------------------------------------------------------------------- #


def _safetensor_files(path: Path) -> dict[str, Path]:
    """Map checkpoint key -> shard file for a safetensors file or directory."""
    from safetensors import safe_open

    path = Path(path)
    if path.is_file():
        with safe_open(str(path), framework="pt") as f:
            return {k: path for k in f.keys()}
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if indexes:
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        return {k: path / v for k, v in weight_map.items()}
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights in {path}")
    out: dict[str, Path] = {}
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            out.update({k: file for k in f.keys()})
    return out


def resolve_keys(model_keys: Iterable[str], ckpt_keys: Iterable[str]) -> dict[str, str]:
    """Match model state-dict keys to checkpoint keys.

    Exact matches first, then a unique-suffix match, which absorbs wrapper
    prefixes such as ``model.`` / ``language_model.`` that differ between the
    library version that saved a checkpoint and the one loading it.
    """
    ckpt = set(ckpt_keys)
    mapping: dict[str, str] = {}
    pending = []
    for key in model_keys:
        if key in ckpt:
            mapping[key] = key
        else:
            pending.append(key)
    if not pending:
        return mapping

    by_tail: dict[str, list[str]] = {}
    for c in ckpt:
        parts = c.split(".")
        for i in range(len(parts)):
            by_tail.setdefault(".".join(parts[i:]), []).append(c)
    used = set(mapping.values())
    for key in pending:
        parts = key.split(".")
        for i in range(len(parts)):
            cands = [c for c in by_tail.get(".".join(parts[i:]), []) if c not in used]
            if len(cands) == 1:
                mapping[key] = cands[0]
                used.add(cands[0])
                break
            if len(cands) > 1:
                break  # ambiguous; a shorter tail would only be worse
    return mapping


class SafetensorsSource:
    """Reads tensors lazily (mmap) from a safetensors file/shard directory."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._files = _safetensor_files(self.path)
        self._handles: dict[Path, object] = {}
        self._lock = threading.Lock()
        self.mapping: dict[str, str] = {}

    def keys(self) -> set[str]:
        return set(self._files)

    def bind(self, model_keys: Iterable[str]) -> list[str]:
        """Resolve model keys; return the ones with no checkpoint tensor."""
        model_keys = list(model_keys)
        self.mapping = resolve_keys(model_keys, self._files)
        return [k for k in model_keys if k not in self.mapping]

    def _handle(self, file: Path):
        from safetensors import safe_open

        h = self._handles.get(file)
        if h is None:
            h = safe_open(str(file), framework="pt", device="cpu")
            self._handles[file] = h
        return h

    def read(self, model_keys: Iterable[str]) -> dict[str, torch.Tensor]:
        out = {}
        with self._lock:
            for key in model_keys:
                ck = self.mapping.get(key)
                if ck is None:
                    continue
                out[key] = self._handle(self._files[ck]).get_tensor(ck)
        return out

    def nbytes(self, model_keys: Iterable[str]) -> int:
        with self._lock:
            total = 0
            for key in model_keys:
                ck = self.mapping.get(key)
                if ck is not None:
                    sl = self._handle(self._files[ck]).get_slice(ck)
                    shape = sl.get_shape()
                    n = 1
                    for s in shape:
                        n *= s
                    total += n * _dtype_size(sl.get_dtype())
            return total


def _dtype_size(name: str) -> int:
    return {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2}.get(name, 1)


class LayerCacheSource:
    """A pre-split cache: one safetensors file per streamed unit (AirLLM style).

    Files hold tensors under their *model* key, optionally fp8-compressed with a
    per-tensor scale stored under ``<key>::scale``.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != CACHE_FORMAT:
            raise ValueError(f"unsupported layer cache format in {self.path}")
        self._file_of: dict[str, str] = self.manifest["keys"]
        self._handles: dict[str, object] = {}
        self._lock = threading.Lock()

    @property
    def compress(self) -> str:
        return self.manifest.get("compress", "none")

    def keys(self) -> set[str]:
        return set(self._file_of)

    def bind(self, model_keys: Iterable[str]) -> list[str]:
        return [k for k in model_keys if k not in self._file_of]

    def _handle(self, name: str):
        from safetensors import safe_open

        h = self._handles.get(name)
        if h is None:
            h = safe_open(str(self.path / name), framework="pt", device="cpu")
            self._handles[name] = h
        return h

    def read(self, model_keys: Iterable[str]) -> dict[str, torch.Tensor]:
        raw: dict[str, torch.Tensor] = {}
        scales: dict[str, torch.Tensor] = {}
        by_file: dict[str, list[str]] = {}
        for key in model_keys:
            name = self._file_of.get(key)
            if name is not None:
                by_file.setdefault(name, []).append(key)
        for name, keys in by_file.items():
            if name.startswith("unit_"):
                # One sequential read per block (4-6 GB/s on Apple SSDs). mmap'd
                # get_tensor is lazy and pays for it later in page faults at a
                # fraction of the throughput, because each step streams more than
                # fits in the page cache.
                from safetensors.torch import load as load_bytes

                with open(self.path / name, "rb", buffering=0) as fh:
                    tensors = load_bytes(fh.read())
                get = tensors.__getitem__
            else:
                with self._lock:
                    h = self._handle(name)
                get = h.get_tensor
            for key in keys:
                t = get(key)
                raw[key] = t
                if FP8 is not None and t.dtype == FP8:
                    scales[key] = get(key + SCALE_SUFFIX)
        if not scales:
            return raw

        def dequant(key: str) -> tuple[str, torch.Tensor]:
            # fp8 -> bf16 is exact; scaling in bf16 avoids a 4-byte/param temporary
            return key, raw[key].to(torch.bfloat16).mul_(scales[key].to(torch.bfloat16))

        # Dequantisation is the per-block bottleneck on CPU (~0.6 s for a 22B DiT
        # block single-threaded); torch releases the GIL, so a pool gives ~3.5x.
        raw.update(_dequant_pool().map(dequant, list(scales)))
        return raw


_POOL: ThreadPoolExecutor | None = None


def _dequant_pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=max(2, min(6, (os.cpu_count() or 4) - 2)),
                                   thread_name_prefix="airstream-dequant")
    return _POOL


def open_source(path: str | os.PathLike):
    path = Path(path)
    if (path / "manifest.json").exists():
        return LayerCacheSource(path)
    return SafetensorsSource(path)


# --------------------------------------------------------------------------- #
# Model surgery helpers
# --------------------------------------------------------------------------- #


def find_units(model: nn.Module, min_blocks: int = 2) -> list[tuple[str, nn.Module]]:
    """Streamable units = the children of every ``nn.ModuleList`` of repeated blocks."""
    units: list[tuple[str, nn.Module]] = []
    claimed: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList) or len(mod) < min_blocks:
            continue
        if any(name.startswith(c + ".") for c in claimed):
            continue  # nested list inside an already-streamed block
        if not all(any(True for _ in child.parameters()) for child in mod):
            continue
        claimed.append(name)
        units.extend((f"{name}.{i}", child) for i, child in enumerate(mod))
    return units


def _persistent_keys(model: nn.Module) -> list[str]:
    """State-dict keys including tied duplicates (params + persistent buffers)."""
    keys = [n for n, _ in model.named_parameters(remove_duplicate=False)]
    for mod_name, mod in model.named_modules(remove_duplicate=False):
        for b in mod._buffers:
            if b not in mod._non_persistent_buffers_set and mod._buffers[b] is not None:
                keys.append(f"{mod_name}.{b}" if mod_name else b)
    return list(dict.fromkeys(keys))


def _set_tensor(root: nn.Module, key: str, value: torch.Tensor) -> None:
    mod_path, _, name = key.rpartition(".")
    mod = root.get_submodule(mod_path) if mod_path else root
    if name in mod._parameters:
        mod._parameters[name] = nn.Parameter(value, requires_grad=False)
    elif name in mod._buffers:
        mod._buffers[name] = value
    else:
        raise KeyError(key)


def _get_tensor(root: nn.Module, key: str) -> torch.Tensor:
    mod_path, _, name = key.rpartition(".")
    mod = root.get_submodule(mod_path) if mod_path else root
    t = mod._parameters.get(name)
    return t if t is not None else mod._buffers[name]


def _keep_fp32(model: nn.Module) -> list[str]:
    return list(getattr(model, "_keep_in_fp32_modules", None) or [])


# --------------------------------------------------------------------------- #
# The streamer
# --------------------------------------------------------------------------- #


@dataclass
class StreamStats:
    bytes_read: int = 0
    io_seconds: float = 0.0
    loads: int = 0
    prefetch_hits: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def gbps(self) -> float:
        return self.bytes_read / 1e9 / self.io_seconds if self.io_seconds else 0.0


UnitCallback = Callable[[int, int, str], None]


class AirStream:
    """Attach disk-streaming to a model whose weights live on ``meta``.

    >>> with accelerate.init_empty_weights():
    ...     model = Model.from_config(cfg)
    >>> stream = AirStream(model, open_source(weights_dir), device="mps")
    >>> model(x)  # blocks are loaded/evicted on the fly
    """

    def __init__(
        self,
        model: nn.Module,
        source,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
        prefetch: bool = True,
        on_unit: UnitCallback | None = None,
        strict: bool = True,
    ):
        self.model = model
        self.source = source
        self.device = torch.device(device)
        self.dtype = dtype
        self.on_unit = on_unit
        self.stats = StreamStats()
        self._fp32 = _keep_fp32(model)

        self.units = find_units(model)
        if not self.units:
            raise ValueError("no repeated blocks found to stream")
        self._unit_index = {id(m): i for i, (_, m) in enumerate(self.units)}

        all_keys = _persistent_keys(model)
        # Tied weights: an alias may be absent from the checkpoint as long as one
        # member of its tie group is present.
        self._aliases = self._tie_groups(model)
        missing = missing_weights(model, source)
        if missing and strict:
            sample = ", ".join(missing[:8])
            raise KeyError(f"{len(missing)} weights missing from checkpoint (e.g. {sample})")

        prefixes = [n + "." for n, _ in self.units]
        self._unit_keys: list[list[str]] = [[] for _ in self.units]
        self._resident_keys: list[str] = []
        for key in all_keys:
            for i, p in enumerate(prefixes):
                if key.startswith(p):
                    self._unit_keys[i].append(key)
                    break
            else:
                self._resident_keys.append(key)

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airstream")
        self._prefetch_enabled = prefetch
        self._pending: dict[int, Future] = {}
        self._next: dict[int, int] = {}
        self._last: int | None = None
        self._loaded: set[int] = set()
        self._hooks = []

        self._materialize(self._resident_keys)
        for _, mod in self.units:
            self._hooks.append(mod.register_forward_pre_hook(self._pre_hook))
            self._hooks.append(mod.register_forward_hook(self._post_hook))

    # -- tie handling ------------------------------------------------------- #
    @staticmethod
    def _tie_groups(model: nn.Module) -> dict[str, tuple[str, ...]]:
        """Weight-tie groups: shared Parameter objects plus declared ties.

        ``init_empty_weights`` re-creates parameters on ``meta``, which silently
        unties e.g. ``lm_head``/``embed_tokens``, so the transformers-style
        ``_tied_weights_keys = {alias: source}`` declarations are honoured too.
        """
        by_id: dict[int, list[str]] = {}
        for n, p in model.named_parameters(remove_duplicate=False):
            by_id.setdefault(id(p), []).append(n)
        pairs: list[tuple[str, str]] = []
        for names in by_id.values():
            pairs += [(names[0], n) for n in names[1:]]
        for prefix, mod in model.named_modules():
            declared = getattr(mod, "_tied_weights_keys", None)
            if isinstance(declared, dict):
                p = prefix + "." if prefix else ""
                pairs += [(p + src, p + alias) for alias, src in declared.items()]
        groups: dict[str, set[str]] = {}
        for a, b in pairs:
            merged = groups.get(a, {a}) | groups.get(b, {b})
            for n in merged:
                groups[n] = merged
        return {n: tuple(sorted(g - {n})) for n, g in groups.items()}

    # -- loading ------------------------------------------------------------ #
    def _target_dtype(self, key: str, t: torch.Tensor) -> torch.dtype:
        if not t.is_floating_point():
            return t.dtype
        if any(f".{m}." in f".{key}" or key.startswith(m + ".") for m in self._fp32):
            return torch.float32
        return self.dtype

    def _read(self, keys: list[str]) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        tensors = self.source.read(keys)
        out = {}
        for k, t in tensors.items():
            self.stats.bytes_read += t.numel() * t.element_size()
            t = t.to(self._target_dtype(k, t))
            if self.device.type == "cuda":
                t = t.pin_memory()
            out[k] = t
        self.stats.io_seconds += time.perf_counter() - t0
        return out

    def _materialize(self, keys: list[str], tensors: dict[str, torch.Tensor] | None = None) -> None:
        if tensors is None:
            tensors = self._pool.submit(self._read, keys).result()
        for key in keys:
            t = tensors.get(key)
            if t is None:
                for alias in self._aliases.get(key, ()):
                    if alias in tensors:
                        t = tensors[alias]
                        break
            if t is None:
                continue  # non-strict: left on meta / initialised value
            dev = t.to(self.device, non_blocking=self.device.type == "cuda")
            _set_tensor(self.model, key, dev)
            for alias in self._aliases.get(key, ()):
                _set_tensor(self.model, alias, _get_tensor(self.model, key))

    def _evict(self, keys: list[str]) -> None:
        for key in keys:
            t = _get_tensor(self.model, key)
            _set_tensor(self.model, key, torch.empty(t.shape, dtype=t.dtype, device="meta"))

    def _schedule_prefetch(self, idx: int) -> None:
        if not self._prefetch_enabled:
            return
        nxt = self._next.get(idx, idx + 1 if idx + 1 < len(self.units) else None)
        if nxt is None or nxt in self._pending or nxt in self._loaded:
            return
        self._pending[nxt] = self._pool.submit(self._read, self._unit_keys[nxt])

    # -- hooks -------------------------------------------------------------- #
    def _pre_hook(self, module, args):
        idx = self._unit_index[id(module)]
        if self._last is not None and self._last != idx:
            self._next[self._last] = idx
        self._last = idx
        if idx not in self._loaded:
            fut = self._pending.pop(idx, None)
            if fut is not None:
                self.stats.prefetch_hits += 1
                tensors = fut.result()
            else:
                tensors = self._pool.submit(self._read, self._unit_keys[idx]).result()
            self._materialize(self._unit_keys[idx], tensors)
            self._loaded.add(idx)
            self.stats.loads += 1
        self._schedule_prefetch(idx)
        if self.on_unit:
            self.on_unit(idx, len(self.units), self.units[idx][0])

    def _post_hook(self, module, args, output):
        idx = self._unit_index[id(module)]
        self._evict(self._unit_keys[idx])
        self._loaded.discard(idx)
        return output

    # -- lifecycle ---------------------------------------------------------- #
    def release(self) -> None:
        """Drop resident weights and hooks; the model returns to ``meta``."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for fut in self._pending.values():
            fut.cancel()
        self._pending.clear()
        self._evict(self._resident_keys)
        for idx in list(self._loaded):
            self._evict(self._unit_keys[idx])
        self._loaded.clear()
        self._pool.shutdown(wait=True)
        _empty_cache(self.device)

    def resident_bytes(self) -> int:
        total = 0
        for key in self._resident_keys:
            t = _get_tensor(self.model, key)
            if t.device.type != "meta":
                total += t.numel() * t.element_size()
        return total


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def missing_weights(model: nn.Module, source) -> list[str]:
    """Model tensors the checkpoint can't supply (tied aliases excused)."""
    missing = source.bind(_persistent_keys(model))
    ties = AirStream._tie_groups(model)
    return [k for k in missing if not any(a not in missing for a in ties.get(k, ()))]


# --------------------------------------------------------------------------- #
# Layer cache builder
# --------------------------------------------------------------------------- #


def _quantize_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row float8 (e4m3) quantisation."""
    w = t.to(torch.float32)
    scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    return (w / scale).to(FP8), scale


def build_layer_cache(
    model: nn.Module,
    weights: str | os.PathLike,
    out_dir: str | os.PathLike,
    compress: str = "none",
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Split ``weights`` into one file per streamed unit (+ one for residents).

    ``compress="fp8"`` stores Linear weights as float8 with per-row scales,
    halving the bytes AirStream has to read per block.
    """
    from safetensors.torch import save_file

    if compress not in ("none", "fp8"):
        raise ValueError("compress must be 'none' or 'fp8'")
    if compress == "fp8" and FP8 is None:
        raise RuntimeError("this torch build has no float8 support")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = SafetensorsSource(weights)
    all_keys = _persistent_keys(model)
    missing = set(source.bind(all_keys))
    units = find_units(model)
    prefixes = [n + "." for n, _ in units]
    groups: list[list[str]] = [[] for _ in units]
    resident: list[str] = []
    for key in all_keys:
        if key in missing:
            continue
        for i, p in enumerate(prefixes):
            if key.startswith(p):
                groups[i].append(key)
                break
        else:
            resident.append(key)

    fp32 = _keep_fp32(model)
    # Only Linear weights are quantised; embeddings, norms, convs and
    # modulation tables stay in their checkpoint precision.
    linear = {f"{n}.weight" for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    file_of: dict[str, str] = {}
    jobs = [("resident.safetensors", resident)] + [
        (f"unit_{i:04d}.safetensors", keys) for i, keys in enumerate(groups)
    ]
    seen_ckpt: dict[str, str] = {}
    for n, (name, keys) in enumerate(jobs):
        tensors = {}
        for key, t in source.read(keys).items():
            ck = source.mapping[key]
            if ck in seen_ckpt:  # tied duplicate: stored once, re-tied on load
                continue
            seen_ckpt[ck] = key
            keep = any(key.startswith(m + ".") or f".{m}." in key for m in fp32)
            if compress == "fp8" and not keep and key in linear and t.dim() == 2 and t.numel() >= 4096:
                q, s = _quantize_fp8(t)
                tensors[key] = q
                tensors[key + SCALE_SUFFIX] = s
            else:
                tensors[key] = t.contiguous()
            file_of[key] = name
        if tensors:
            save_file(tensors, str(out / name))
        if on_progress:
            on_progress(n + 1, len(jobs))

    manifest = {"format": CACHE_FORMAT, "compress": compress, "keys": file_of, "units": len(units)}
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out

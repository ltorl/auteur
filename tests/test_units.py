import pytest

from auteur.llm import extract_json, rank_image_models, rank_text_models
from auteur.render.base import RESOLUTIONS, frames_for
from auteur.schemas import ShotPlan, slugify


def test_frames_are_8k_plus_1():
    for sec in (0.1, 1, 2.5, 3, 5, 7.3, 10, 30):
        f = frames_for(sec)
        assert f % 8 == 1 and 9 <= f <= 241
    assert frames_for(5) == 121


def test_resolutions_fit_two_stage_grid():
    for aspect in RESOLUTIONS.values():
        for w, h in aspect.values():
            assert w % 64 == 0 and h % 64 == 0


def test_model_ranking_prefers_newest_full_size():
    ids = ["gpt-5.4-mini", "gpt-5.6-terra", "gpt-5.6-codex", "gpt-5.5", "gpt-image-2", "gpt-image-1"]
    assert rank_text_models(ids)[0] == "gpt-5.6-terra"
    assert "gpt-image-2" not in rank_text_models(ids)
    assert rank_image_models(ids)[0] == "gpt-image-2"


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


def test_shot_duration_clamped_and_slugs():
    s = ShotPlan(id="S01", title="t", duration=42, shot_size="CU", camera="static", action="a",
                 keyframe_prompt="k", motion_prompt="m")
    assert s.duration == 10
    assert slugify("Ünïcode & Spaces!!") == "n-code-spaces"


torch = pytest.importorskip("torch")


def test_resolve_keys_prefix_and_ambiguity():
    from auteur.render.airstream import resolve_keys

    m = resolve_keys(["layers.0.w", "layers.1.w", "head.weight"],
                     ["model.layers.0.w", "model.layers.1.w", "emb.weight", "other.weight"])
    assert m == {"layers.0.w": "model.layers.0.w", "layers.1.w": "model.layers.1.w"}


def _toy():
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(64, 64)

        def forward(self, x):
            return x + torch.tanh(self.lin(x))

    class Toy(nn.Module):
        _tied_weights_keys = {"head.weight": "emb.weight"}

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 64)
            self.blocks = nn.ModuleList([Block() for _ in range(5)])
            self.head = nn.Linear(64, 10, bias=False)
            self.head.weight = self.emb.weight

        def forward(self, i):
            x = self.emb(i)
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    return Toy


@pytest.mark.parametrize("compress", [None, "none", "fp8"])
def test_airstream_matches_dense(tmp_path, compress):
    from accelerate import init_empty_weights
    from safetensors.torch import save_file

    from auteur.render.airstream import AirStream, build_layer_cache, open_source

    Toy = _toy()
    torch.manual_seed(0)
    ref = Toy().eval()
    sd = {f"model.{k}": v.contiguous() for k, v in ref.state_dict().items() if k != "head.weight"}
    keys = list(sd)
    save_file({k: sd[k] for k in keys[:5]}, tmp_path / "a.safetensors")
    save_file({k: sd[k] for k in keys[5:]}, tmp_path / "b.safetensors")
    src = tmp_path
    if compress:
        with init_empty_weights():
            build_layer_cache(Toy(), tmp_path, tmp_path / "cache", compress=compress)
        src = tmp_path / "cache"
    with init_empty_weights():
        m = Toy().eval()
    s = AirStream(m, open_source(src), "cpu", dtype=torch.float32)
    x = torch.tensor([[1, 2, 3]])
    want = ref(x)
    for _ in range(2):  # second pass exercises learned prefetch order
        got = m(x)
    err = ((got - want).norm() / want.norm()).item()
    assert err < (0.05 if compress == "fp8" else 1e-7)
    assert s.stats.loads == 10 and s.stats.prefetch_hits >= 8
    assert all(p.device.type == "meta" for b in m.blocks for p in b.parameters())
    s.release()
    assert m.emb.weight.device.type == "meta"


def test_weights_progress_and_completion(tmp_path):
    import json

    from auteur import weights as W

    W.save_plan(tmp_path, {"transformer/a.safetensors": 10, "vae/b.safetensors": 30})
    assert not W.present(tmp_path)
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "a.safetensors").write_bytes(b"x" * 10)
    part = tmp_path / ".cache" / "huggingface" / "download" / "vae"
    part.mkdir(parents=True)
    (part / "b.safetensors.abc.incomplete").write_bytes(b"x" * 15)
    p = W.progress(tmp_path)
    assert (p.done_files, p.total_files, p.done_bytes, p.total_bytes) == (1, 2, 25, 40)
    assert not W.present(tmp_path)  # a partial download is never "present"
    (tmp_path / W.DONE_FILE).write_text(json.dumps({"files": ["transformer/a.safetensors"]}))
    assert W.present(tmp_path)

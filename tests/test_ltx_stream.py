"""Layer-streamed LTX pipeline == in-memory pipeline, on a tiny random LTX-2 model.

Uses the same miniature components as diffusers' own LTX2 tests, saved to disk
in the diffusers repo layout that ``Lightricks/LTX-2.5-Diffusers`` uses.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.slow


def tiny_components():
    from diffusers import (
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
        FlowMatchEulerDiscreteScheduler,
        LTX2VideoTransformer3DModel,
    )
    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-gemma3")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-gemma3")
    hidden = text_encoder.config.text_config.hidden_size
    torch.manual_seed(0)
    transformer = LTX2VideoTransformer3DModel(
        in_channels=4, out_channels=4, patch_size=1, patch_size_t=1,
        num_attention_heads=2, attention_head_dim=8, cross_attention_dim=16,
        audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2,
        audio_attention_head_dim=4, audio_cross_attention_dim=8, num_layers=3,
        qk_norm="rms_norm_across_heads", caption_channels=hidden,
        rope_double_precision=False, rope_type="split",
    )
    torch.manual_seed(0)
    connectors = LTX2TextConnectors(
        caption_channels=hidden,
        text_proj_in_factor=text_encoder.config.text_config.num_hidden_layers + 1,
        video_connector_num_attention_heads=4, video_connector_attention_head_dim=8,
        video_connector_num_layers=2, video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=4, audio_connector_attention_head_dim=8,
        audio_connector_num_layers=2, audio_connector_num_learnable_registers=None,
        connector_rope_base_seq_len=32, rope_theta=10000.0, rope_double_precision=False,
        causal_temporal_positioning=False, rope_type="split",
    )
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8,),
        decoder_block_out_channels=(8,), layers_per_block=(1,), decoder_layers_per_block=(1, 1),
        spatio_temporal_scaling=(True,), decoder_spatio_temporal_scaling=(True,),
        decoder_inject_noise=(False, False), downsample_type=("spatial",),
        upsample_residual=(False,), upsample_factor=(1,), timestep_conditioning=False,
        patch_size=1, patch_size_t=1, encoder_causal=True, decoder_causal=False,
    )
    torch.manual_seed(0)
    audio_vae = AutoencoderKLLTX2Audio(
        base_channels=4, output_channels=2, ch_mult=(1,), num_res_blocks=1, attn_resolutions=None,
        in_channels=2, resolution=32, latent_channels=2, norm_type="pixel", causality_axis="height",
        dropout=0.0, mid_block_add_attention=False, sample_rate=16000, mel_hop_length=160,
        is_causal=True, mel_bins=8,
    )
    torch.manual_seed(0)
    vocoder = LTX2Vocoder(
        in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins,
        hidden_channels=32, out_channels=2, upsample_kernel_sizes=[4, 4], upsample_factors=[2, 2],
        resnet_kernel_sizes=[3], resnet_dilations=[[1, 3, 5]], leaky_relu_negative_slope=0.1,
        output_sampling_rate=16000,
    )
    torch.manual_seed(0)
    upsampler = LTX2LatentUpsamplerModel(in_channels=4, mid_channels=32, num_blocks_per_stage=1)
    return dict(
        transformer=transformer, vae=vae, audio_vae=audio_vae, scheduler=FlowMatchEulerDiscreteScheduler(),
        text_encoder=text_encoder, tokenizer=tokenizer, connectors=connectors, vocoder=vocoder,
        processor=None, prompt_enhancer=None, duration_head=None,
    ), upsampler


@pytest.fixture(scope="module")
def tiny_repo(tmp_path_factory):
    from diffusers import LTX2ImageToVideoPipeline

    comps, upsampler = tiny_components()
    root = tmp_path_factory.mktemp("ltx-tiny")
    pipe = LTX2ImageToVideoPipeline(**comps).to(torch.bfloat16)
    pipe.save_pretrained(root)
    upsampler.to(torch.bfloat16).save_pretrained(root / "latent_upsampler")
    # Reference = the stock diffusers load path, which is what streaming must reproduce.
    return root, LTX2ImageToVideoPipeline.from_pretrained(root, dtype=torch.bfloat16)


def _job(tmp_path, image=None):
    from auteur.render.base import ShotJob

    return ShotJob("S01", "a robot dancing in the rain", tmp_path / "s01.mp4", seconds=0.33,
                   width=32, height=32, seed=7, image=image)


def _reference(pipe, job, image, **kw):
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

    pipe.set_progress_bar_config(disable=True)
    if image is not None:
        kw["image"] = image
    else:
        from diffusers import LTX2Pipeline

        pipe = LTX2Pipeline(**pipe.components)
        pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        return pipe(
            prompt=job.prompt, width=job.width, height=job.height, num_frames=job.num_frames,
            frame_rate=job.fps, sigmas=DISTILLED_SIGMA_VALUES, guidance_scale=1.0,
            audio_guidance_scale=1.0, stg_scale=0.0, audio_stg_scale=0.0, modality_scale=1.0,
            audio_modality_scale=1.0, generator=torch.Generator("cpu").manual_seed(job.seed),
            output_type="np", return_dict=False, **kw,
        )


@pytest.mark.parametrize("with_image", [True, False])
def test_streamed_matches_in_memory(tiny_repo, tmp_path, with_image):
    from PIL import Image

    from auteur.render.ltx import LTXStreamEngine

    root, ref_pipe = tiny_repo
    image = Image.fromarray((np.random.default_rng(0).random((32, 32, 3)) * 255).astype("uint8"))
    job = _job(tmp_path)
    want_v, want_a = _reference(ref_pipe, job, image if with_image else None)

    events = []
    eng = LTXStreamEngine(root, device="cpu")
    eng._emit = events.append
    eng.load()
    (embeds,) = eng.encode([job.prompt])
    got_v, got_a = eng._call(job, embeds, image if with_image else None, 32, 32,
                             torch.Generator("cpu").manual_seed(job.seed), output_type="np")
    np.testing.assert_allclose(np.asarray(got_v), np.asarray(want_v), atol=0)
    torch.testing.assert_close(got_a.float(), want_a.float())

    units = [e for e in events if e.kind == "unit"]
    assert {e.text for e in units} == {"DiT", "text encoder", "connectors"}
    assert max(e.n for e in units if e.text == "DiT") == 3
    # every DiT block went back to meta after the call
    assert all(p.device.type == "meta" for b in eng.pipe.transformer.transformer_blocks for p in b.parameters())
    eng.close()


def test_render_writes_mp4_and_fp8_cache(tiny_repo, tmp_path):
    from auteur.render import media
    from auteur.render.ltx import STREAMED, LTXStreamEngine, build_caches, verify

    root, _ = tiny_repo
    assert verify(root) == {c: [] for c in STREAMED}
    build_caches(root, tmp_path / "cache", "fp8")
    eng = LTXStreamEngine(root, device="cpu", cache_root=tmp_path / "cache", compress="fp8")
    (out,) = eng.render([_job(tmp_path)], lambda e: None)
    info = media.probe(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (int(v["width"]), int(v["height"])) == (32, 32)
    assert media.has_audio(out)
    eng.close()


def test_two_stage(tiny_repo, tmp_path):
    from PIL import Image

    from auteur.render.ltx import LTXStreamEngine

    root, _ = tiny_repo
    img = tmp_path / "key.png"
    Image.new("RGB", (96, 64), (200, 80, 40)).save(img)
    job = _job(tmp_path, image=img)
    job.width = job.height = 64
    eng = LTXStreamEngine(root, device="cpu", two_stage=True)
    (out,) = eng.render([job], lambda e: None)
    assert out.exists() and out.stat().st_size > 0
    eng.close()

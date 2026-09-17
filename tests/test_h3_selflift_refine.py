from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--cpu"]

spec = importlib.util.spec_from_file_location(
    "director_h3_selflift_tests",
    PLUGIN_ROOT / "__init__.py",
    submodule_search_locations=[str(PLUGIN_ROOT)],
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
assert spec.loader is not None
spec.loader.exec_module(plugin)

from director_h3_selflift_tests.director import refine_sampling  # noqa: E402
from director_h3_selflift_tests.director.refine_pack import (  # noqa: E402
    pack_refine,
    refine_fingerprint,
)
from director_h3_selflift_tests.nodes.director_refine import (  # noqa: E402
    MiniMaxH3DirectorRefine,
)


def _plan(pack):
    return SimpleNamespace(refine=pack)


def test_refine_node_exposes_h3_selflift_mode_and_settings():
    inputs = MiniMaxH3DirectorRefine.INPUT_TYPES()
    modes = inputs["required"]["mode"][0]
    assert "h3_selflift" in modes
    assert "selflift_transition_step" in inputs["optional"]
    assert "selflift_lowres_scale" in inputs["optional"]
    assert "selflift_upscaler_model" in inputs["optional"]

    config, _, _ = MiniMaxH3DirectorRefine().pack(
        mode="h3_selflift",
        sigmas=torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        selflift_transition_step=5,
        selflift_lowres_scale=0.4,
        selflift_rho=0.6,
        selflift_w_min=0.5,
        selflift_w_max=1.0,
        selflift_upscaler_model="none",
        selflift_highres_tiling=True,
    )
    assert config["mode"] == "h3_selflift"
    assert config["selflift_transition_step"] == 5
    assert config["selflift_lowres_scale"] == 0.4
    assert config["selflift_upscaler_model"] == "none"
    assert config["selflift_highres_tiling"] is True


def test_h3_selflift_refine_receives_first_pass_av_latent():
    samples = {
        "samples": torch.zeros(1, 24, 2, 4, 4),
        "batch_index": [0],
    }
    pack = pack_refine(
        mode="h3_selflift",
        sigmas=torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        selflift_upscaler_model="none",
    )
    calls = []

    def fake_selflift(config, **kwargs):
        calls.append(kwargs)
        assert kwargs["latent"]["samples"] is samples["samples"]
        return {**kwargs["latent"], "samples": kwargs["latent"]["samples"] + 1}

    seg = SimpleNamespace(index=0, task_key="t2v")
    with patch.object(refine_sampling, "sample_h3_selflift", side_effect=fake_selflift):
        result, note = refine_sampling.apply_segment_refine(
            _plan(pack),
            seg,
            samples=samples,
            model=object(),
            vae=object(),
            positive=[[torch.ones(1, 1), {}]],
            negative=[],
            seed=123,
            cfg=1.0,
            first_steps=25,
            sampler_name="res_multistep",
            scheduler="simple",
            shift_video=12.0,
            shift_audio=3.0,
        )

    assert len(calls) == 1
    assert calls[0]["seed"] == 123
    assert torch.equal(result["samples"], samples["samples"] + 1)
    assert "h3_selflift" in note


def test_normal_refine_still_uses_existing_sampler_path():
    samples = {"samples": torch.zeros(1, 24, 2, 4, 4)}
    pack = pack_refine(
        mode="refine",
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )
    calls = []

    def fake_sample(**kwargs):
        calls.append(kwargs)
        return {**kwargs["latent"], "samples": kwargs["latent"]["samples"] + 2}

    with patch.object(refine_sampling, "sample_single_stage", side_effect=fake_sample), patch.object(
        refine_sampling, "sample_h3_selflift", side_effect=AssertionError("SelfLift used for normal refine")
    ):
        result, _ = refine_sampling.apply_segment_refine(
            _plan(pack),
            SimpleNamespace(index=0, task_key="t2v"),
            samples=samples,
            model=object(),
            vae=object(),
            positive=[],
            negative=[],
            seed=1,
            cfg=1.0,
            first_steps=25,
            sampler_name="euler",
            scheduler="simple",
            shift_video=12.0,
            shift_audio=3.0,
        )

    assert len(calls) == 1
    assert torch.equal(result["samples"], samples["samples"] + 2)


def test_h3_settings_are_part_of_refine_cache_fingerprint():
    pack = pack_refine(
        mode="h3_selflift",
        sigmas=torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        selflift_lowres_scale=0.4,
        selflift_upscaler_model="none",
    )
    plan = _plan(pack)
    before = refine_fingerprint(plan)
    pack["selflift_lowres_scale"] = 0.5
    after = refine_fingerprint(plan)
    assert before != after


def test_selflift_upscaler_must_be_a_registered_model_name():
    from director_h3_selflift_tests.director.selflift_sampling import sample_h3_selflift

    config = pack_refine(
        mode="h3_selflift",
        sigmas=torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        selflift_upscaler_model="../../outside.safetensors",
    )
    with patch("folder_paths.get_filename_list", return_value=[]), patch.dict(
        "nodes.NODE_CLASS_MAPPINGS", {}, clear=False
    ):
        try:
            sample_h3_selflift(
                config,
                model=object(),
                vae=object(),
                positive=[],
                negative=[],
                latent={"samples": torch.zeros(1, 24, 2, 4, 4)},
                seed=0,
                cfg=1.0,
            )
        except ValueError as exc:
            assert "latent_upscale_models" in str(exc)
        else:
            raise AssertionError("unregistered SelfLift upscaler was accepted")


def test_selflift_adapter_passes_av_latent_and_schedule_to_node():
    from director_h3_selflift_tests.director import selflift_sampling

    config = pack_refine(
        mode="h3_selflift",
        sigmas=torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
        selflift_transition_step=5,
        selflift_lowres_scale=0.4,
        selflift_rho=0.6,
        selflift_w_min=0.5,
        selflift_w_max=1.0,
        selflift_upscaler_model="none",
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4), "noise_mask": torch.ones(1, 1, 2, 4, 4)}
    calls = []
    model = object()
    model_hires = object()
    shifted_model = object()
    shifted_hires = object()

    class FakeSampler:
        def sample(self, **kwargs):
            calls.append(kwargs)
            return (kwargs["latent_image"],)

    with patch.dict("nodes.NODE_CLASS_MAPPINGS", {"SelfLiftH3Sampler": FakeSampler}, clear=False), patch.object(
        selflift_sampling.KSamplerSelect,
        "execute",
        return_value=(object(),),
    ), patch.object(
        selflift_sampling.MiniMaxH3SigmaShift,
        "execute",
        side_effect=[(shifted_model,), (shifted_hires,)],
    ):
        result = selflift_sampling.sample_h3_selflift(
            config,
            model=model,
            model_hires=model_hires,
            vae=object(),
            positive=[[torch.ones(1, 1), {}]],
            negative=[],
            latent=latent,
            seed=123,
            cfg=1.0,
        )

    assert result is latent
    assert len(calls) == 1
    assert calls[0]["model"] is shifted_model
    assert calls[0]["model_hires"] is shifted_hires
    assert calls[0]["latent_image"] is latent
    assert calls[0]["transition_step"] == 5
    assert calls[0]["lowres_scale"] == 0.4
    assert calls[0]["rho"] == 0.6
    assert calls[0]["sigmas"].shape == (6,)


def test_original_director_keeps_the_default_sampling_stage():
    from director_h3_selflift_tests.nodes import director

    plan = SimpleNamespace(audio_decode_cache={}, global_ref_audios=[])
    frames = torch.zeros(1, 4, 4, 3)
    captured = {}

    def fake_core(_plan, **kwargs):
        captured.update(kwargs)
        return frames, [frames], [{}], "ok", [1], frames, [frames], False

    with patch.object(director, "prepare_director_plan", return_value=plan), patch.object(
        director, "execute_director_plan_core", side_effect=fake_core
    ), patch.object(director, "finalize_director_outputs", return_value=("done",)):
        result = director.MiniMaxH3Director().execute(
            model=object(),
            video_vae=object(),
            audio_vae=object(),
            clip=object(),
            task_type="r2v",
            global_prompt="test",
            frame_rate=24.0,
            width=864,
            height=480,
            ref_max_size=864,
            total_frames=124,
            timeline_data="{}",
        )

    assert result == ("done",)
    assert "sample_stage" not in captured
    assert "sampling_label" not in captured


if __name__ == "__main__":
    names = [name for name in globals() if name.startswith("test_")]
    suite = unittest.TestSuite(unittest.FunctionTestCase(globals()[name]) for name in names)
    unittest.TextTestRunner(verbosity=2).run(suite)

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--cpu"]

if "director_h3_selflift_tests" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "director_h3_selflift_tests",
        PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    assert spec.loader is not None
    spec.loader.exec_module(plugin)

from director_h3_selflift_tests import NODE_CLASS_MAPPINGS  # noqa: E402
from director_h3_selflift_tests.nodes import director_selflift  # noqa: E402


def test_selflift_director_is_r2v_only_and_requires_a_sigma_schedule():
    cls = NODE_CLASS_MAPPINGS["MiniMaxH3DirectorSelfLift"]
    inputs = cls.INPUT_TYPES()

    assert inputs["required"]["task_type"][0] == ["r2v"]
    assert inputs["required"]["sigmas"][0] == "SIGMAS"
    assert "r2v_groups" in inputs["optional"]
    assert "i2v_groups" not in inputs["optional"]
    assert "refine" not in inputs["optional"]
    assert "steps" not in inputs["optional"]
    assert "sampler" not in inputs["optional"]
    assert "scheduler" not in inputs["optional"]


def test_selflift_director_uses_selflift_as_the_only_sampling_stage():
    cls = NODE_CLASS_MAPPINGS["MiniMaxH3DirectorSelfLift"]
    plan = SimpleNamespace(
        segments=[SimpleNamespace(task_key="r2v")],
        audio_decode_cache={},
        global_ref_audios=[],
    )
    latent = {"samples": torch.zeros(1, 24, 2, 4, 4)}
    captured = {}

    def fake_core(plan_arg, **kwargs):
        captured.update(kwargs)
        sampled = kwargs["sample_stage"](
            model=kwargs["model"],
            positive=[[torch.ones(1, 1), {}]],
            negative=[],
            latent=latent,
            seed=kwargs["seed"],
            cfg=kwargs["cfg"],
            steps=25,
            sampler_name="res_multistep",
            scheduler="simple",
            shift_video=kwargs["shift_video"],
            shift_audio=kwargs["shift_audio"],
            sigmas=kwargs["sigmas"],
        )
        assert sampled is latent
        frames = torch.zeros(1, 4, 4, 3)
        return frames, [frames], [{}], "ok", [1], frames, [frames], False

    with patch.object(director_selflift, "prepare_director_plan", return_value=plan) as prepare, patch.object(
        director_selflift, "execute_director_plan_core", side_effect=fake_core
    ), patch.object(
        director_selflift, "finalize_director_outputs", return_value=("done",)
    ), patch.object(
        director_selflift, "sample_h3_selflift", return_value=latent
    ) as selflift:
        result = cls().execute(
            model=object(),
            video_vae=object(),
            audio_vae=object(),
            clip=object(),
            sigmas=torch.tensor([1.0, 0.5, 0.0]),
            task_type="r2v",
            global_prompt="test",
            frame_rate=24.0,
            width=864,
            height=480,
            ref_max_size=864,
            total_frames=124,
            timeline_data="{}",
            selflift_transition_step=2,
            model_hires="hires-model",
        )

    assert result == ("done",)
    assert prepare.call_args.kwargs["task_type"] == "r2v"
    assert plan.sample_fingerprint["backend"] == "h3_selflift"
    assert plan.sample_fingerprint["transition_step"] == 2
    assert captured["sampling_label"].startswith("SelfLift")
    config = selflift.call_args.args[0]
    call = selflift.call_args.kwargs
    assert call["latent"] is latent
    assert call["vae"] is not None
    assert config["selflift_transition_step"] == 2
    assert call["model_hires"] == "hires-model"


def test_selflift_director_rejects_non_r2v_plan_before_sampling():
    plan = SimpleNamespace(segments=[SimpleNamespace(task_key="v2v")])

    try:
        director_selflift.ensure_r2v_plan(plan)
    except ValueError as exc:
        assert "R2V" in str(exc)
    else:
        raise AssertionError("non-R2V plan was accepted")


if __name__ == "__main__":
    import unittest

    tests = [name for name in globals() if name.startswith("test_")]
    suite = unittest.TestSuite(unittest.FunctionTestCase(globals()[name]) for name in tests)
    unittest.TextTestRunner(verbosity=2).run(suite)

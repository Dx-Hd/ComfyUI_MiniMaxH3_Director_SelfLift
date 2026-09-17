"""R2V Director whose only sampling stage is SelfLiftH3Sampler."""

from __future__ import annotations

from ..director.executor_core import execute_director_plan_core
from ..director.h3_latent_upscale import list_h3_latent_upscale_models
from ..director.refine_pack import (
    DEFAULT_SELFLIFT_LOWRES_SCALE,
    DEFAULT_SELFLIFT_RHO,
    DEFAULT_SELFLIFT_TRANSITION_STEP,
    DEFAULT_SELFLIFT_W_MAX,
    DEFAULT_SELFLIFT_W_MIN,
    pack_refine,
)
from ..director.selflift_sampling import sample_h3_selflift
from .director_common import (
    director_perf_inputs,
    finalize_director_outputs,
    prepare_director_plan,
    timeline_required_inputs,
)

_CATEGORY = "MiniMaxH3"


def _selflift_upscaler_choices():
    names = [
        name
        for name in list_h3_latent_upscale_models()
        if name and not str(name).startswith("(")
    ]
    return ["none", *names]


def selflift_timeline_required_inputs() -> dict:
    inputs = timeline_required_inputs()
    inputs["task_type"] = (["r2v"], {"default": "r2v", "tooltip": "固定为 R2V 素材组。"})
    prompt_meta = dict(inputs["global_prompt"][1])
    prompt_meta["default"] = "A cinematic scene with natural motion and synchronized ambience"
    inputs["global_prompt"] = ("STRING", prompt_meta)
    frames_meta = dict(inputs["total_frames"][1])
    frames_meta["default"] = 124
    inputs["total_frames"] = ("INT", frames_meta)
    return inputs


def ensure_r2v_plan(plan) -> None:
    invalid = [seg.task_key for seg in plan.segments if seg.task_key != "r2v"]
    if invalid:
        raise ValueError(
            "MiniMax H3 Director · SelfLift 仅支持 R2V 素材组；"
            f"当前计划包含：{', '.join(sorted(set(invalid)))}。"
        )


class MiniMaxH3DirectorSelfLift:
    """Director material-group UI with SelfLift as its single sampling stage."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 UNET。"}),
                "video_vae": ("VAE", {"tooltip": "MiniMax H3 video VAE。"}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE。"}),
                "clip": ("CLIP", {"tooltip": "CLIPLoader type=minimax。"}),
                "sigmas": (
                    "SIGMAS",
                    {
                        "forceInput": True,
                        "tooltip": "SelfLift 的完整单程噪声表；建议接 H3 MODEL 的 8 步 BasicScheduler + ExtendIntermediateSigmas。",
                    },
                ),
                **selflift_timeline_required_inputs(),
            },
            "optional": {
                "r2v_groups": (
                    "MMX_DIR_GROUP",
                    {"tooltip": "外接 Reference to Video 素材组；接线后优先于界面卡片。"},
                ),
                "model_hires": (
                    "MODEL",
                    {"tooltip": "可选：高分辨率续采模型；不接则全程使用主模型。"},
                ),
                "bd_grp_selflift": ("BDGROUP", {"default": "SelfLift 采样"}),
                "selflift_upscaler_model": (
                    _selflift_upscaler_choices(),
                    {"default": "none", "tooltip": "H3 latent 放大权重；none 使用 SelfLift-zero。"},
                ),
                "selflift_transition_step": (
                    "INT",
                    {"default": DEFAULT_SELFLIFT_TRANSITION_STEP, "min": 1, "max": 10000},
                ),
                "selflift_lowres_scale": (
                    "FLOAT",
                    {"default": DEFAULT_SELFLIFT_LOWRES_SCALE, "min": 0.25, "max": 1.0, "step": 0.05},
                ),
                "selflift_rho": (
                    "FLOAT",
                    {"default": DEFAULT_SELFLIFT_RHO, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "selflift_w_min": (
                    "FLOAT",
                    {"default": DEFAULT_SELFLIFT_W_MIN, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "selflift_w_max": (
                    "FLOAT",
                    {"default": DEFAULT_SELFLIFT_W_MAX, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "selflift_highres_tiling": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "SelfLift 高分辨率阶段自动空间分块。"},
                ),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                **director_perf_inputs(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **_kwargs):
        if input_types is not None:
            for name, want in {
                "model": "MODEL",
                "model_hires": "MODEL",
                "video_vae": "VAE",
                "audio_vae": "VAE",
                "clip": "CLIP",
                "sigmas": "SIGMAS",
            }.items():
                got = input_types.get(name)
                if got is not None and got != want:
                    return f"{name}: expected {want}, linked node returns {got}."
        return True

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "source_images", "report")
    OUTPUT_IS_LIST = (True, True, False, False, True, False)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "R2V 素材组导演台。每段直接执行一次 SelfLift Progressive Sampler："
        "低分辨率前缀 → latent 提升 → 高分辨率续采；不运行 Director 旧一采或 Refine 二采。"
    )

    def execute(
        self,
        model,
        video_vae,
        audio_vae,
        clip,
        sigmas,
        task_type,
        global_prompt,
        frame_rate,
        width,
        height,
        ref_max_size,
        total_frames,
        timeline_data,
        unique_id=None,
        r2v_groups=None,
        model_hires=None,
        selflift_upscaler_model="none",
        selflift_transition_step=DEFAULT_SELFLIFT_TRANSITION_STEP,
        selflift_lowres_scale=DEFAULT_SELFLIFT_LOWRES_SCALE,
        selflift_rho=DEFAULT_SELFLIFT_RHO,
        selflift_w_min=DEFAULT_SELFLIFT_W_MIN,
        selflift_w_max=DEFAULT_SELFLIFT_W_MAX,
        selflift_highres_tiling=True,
        cfg=1.0,
        seed=0,
        shift_video=12.0,
        shift_audio=3.0,
        clear_vram_between_segments=True,
        export_source_images=False,
        **kwargs,
    ):
        del task_type, kwargs
        plan = prepare_director_plan(
            timeline_data=timeline_data,
            task_type="r2v",
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            unique_id=unique_id,
            r2v_groups=r2v_groups,
        )
        ensure_r2v_plan(plan)
        config = pack_refine(
            mode="h3_selflift",
            sigmas=sigmas,
            sampler="euler",
            selflift_upscaler_model=selflift_upscaler_model,
            selflift_transition_step=selflift_transition_step,
            selflift_lowres_scale=selflift_lowres_scale,
            selflift_rho=selflift_rho,
            selflift_w_min=selflift_w_min,
            selflift_w_max=selflift_w_max,
            selflift_highres_tiling=selflift_highres_tiling,
        )
        plan.sample_fingerprint = {
            "backend": "h3_selflift",
            "sigmas": [round(float(value), 6) for value in config["sigmas_parsed"]],
            "transition_step": config["selflift_transition_step"],
            "lowres_scale": round(config["selflift_lowres_scale"], 6),
            "rho": round(config["selflift_rho"], 6),
            "w_min": round(config["selflift_w_min"], 6),
            "w_max": round(config["selflift_w_max"], 6),
            "upscaler_model": config["selflift_upscaler_model"],
            "highres_tiling": config["selflift_highres_tiling"],
            "model_hires": model_hires is not None,
            "seed": int(seed),
            "cfg": round(float(cfg), 6),
            "shift_video": round(float(shift_video), 6),
            "shift_audio": round(float(shift_audio), 6),
        }

        def run_selflift(**sample_kwargs):
            return sample_h3_selflift(
                config,
                model=sample_kwargs["model"],
                vae=video_vae,
                positive=sample_kwargs["positive"],
                negative=sample_kwargs["negative"],
                latent=sample_kwargs["latent"],
                seed=sample_kwargs["seed"],
                cfg=sample_kwargs["cfg"],
                shift_video=sample_kwargs["shift_video"],
                shift_audio=sample_kwargs["shift_audio"],
                shift_cache=sample_kwargs.get("shift_cache"),
                on_phase=sample_kwargs.get("on_phase"),
                on_step_preview=sample_kwargs.get("on_step_preview"),
                after_shift=sample_kwargs.get("after_shift"),
                model_hires=model_hires,
                phase_name="sample",
            )

        try:
            results = execute_director_plan_core(
                plan,
                node_id=unique_id,
                model=model,
                vae=video_vae,
                audio_vae=audio_vae,
                clip=clip,
                cfg=cfg,
                seed=seed,
                steps=max(1, len(config["sigmas_parsed"]) - 1),
                sampler="euler",
                scheduler="external_sigmas",
                sigmas=sigmas,
                shift_video=shift_video,
                shift_audio=shift_audio,
                clear_vram_between_segments=clear_vram_between_segments,
                sample_stage=run_selflift,
                sampling_label="SelfLift Progressive Sampler（单程）",
            )
            combined, segments, audios, report, frame_counts, pre_combined, pre_segments, held = results
            outputs = finalize_director_outputs(
                plan,
                combined,
                segments,
                report,
                export_source_images=export_source_images,
                segment_audios=audios,
                segment_frame_counts=frame_counts,
                pre_refine_combined=pre_combined,
                pre_refine_segments=pre_segments,
                block_final_images=held,
                describe_pre_refine=False,
            )
            return outputs[:6]
        finally:
            cache = getattr(plan, "audio_decode_cache", None)
            if isinstance(cache, dict):
                cache.clear()
            for item in getattr(plan, "global_ref_audios", None) or []:
                if getattr(item, "audio_path", ""):
                    item.audio = None

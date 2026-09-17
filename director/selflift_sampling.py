"""SelfLift adapter shared by Director sampling paths."""

from __future__ import annotations

from typing import Any, Callable

import folder_paths
import nodes as comfy_nodes
import torch
from comfy_extras.nodes_custom_sampler import KSamplerSelect
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift


def _unpack(out):
    if hasattr(out, "args"):
        return out.args
    if isinstance(out, (tuple, list)):
        return out
    return (out,)


def _resolve_upscaler_name(config: dict[str, Any]) -> str:
    raw = config.get("selflift_upscaler_model")
    name = "none" if raw is None else str(raw).strip()
    if not name or name.startswith("("):
        return "none"
    if name == "none":
        return name
    try:
        allowed = set(folder_paths.get_filename_list("latent_upscale_models") or [])
    except (AttributeError, KeyError, TypeError):
        allowed = set()
    if name not in allowed:
        raise ValueError(
            "SelfLift 放大模型不在 latent_upscale_models 中：" + repr(name)
        )
    return name


def _resolve_sigmas(config: dict[str, Any]):
    sigmas = config.get("sigmas_tensor")
    if sigmas is None:
        sigmas = config.get("sigmas")
    if sigmas is None:
        raise ValueError("H3 SelfLift 需要外接完整 SIGMAS 噪声表。")
    if torch.is_tensor(sigmas):
        return sigmas.detach().float().cpu().reshape(-1)
    try:
        return torch.tensor([float(value) for value in sigmas], dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("H3 SelfLift 的 SIGMAS 无法解析。") from exc


def _config_value(config: dict[str, Any], name: str, default):
    value = config.get(name)
    return default if value is None else value


def sample_h3_selflift(
    config: dict[str, Any],
    *,
    model,
    vae,
    positive,
    negative,
    latent: dict,
    seed: int,
    cfg: float,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    shift_cache=None,
    on_phase: Callable[[str, float], None] | None = None,
    on_step_preview=None,
    phase_name: str = "refine",
    model_hires=None,
    after_shift=None,
) -> dict:
    """Run SelfLift on a Director-prepared target AV latent."""
    del on_step_preview
    upscaler_model = _resolve_upscaler_name(config)
    sigmas = _resolve_sigmas(config)
    sampler_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("SelfLiftH3Sampler")
    if sampler_cls is None:
        raise RuntimeError(
            "h3_selflift 需要已加载 comfyui-SelfLift 的 SelfLiftH3Sampler 节点。"
        )

    sampler_name = str(config.get("sampler") or "euler").strip().lower()
    if sampler_name != "euler":
        raise ValueError("h3_selflift 只支持 Euler 采样器。")

    if shift_cache is not None:
        model_use = shift_cache.get(model, shift_video, shift_audio)
        model_hires_use = (
            shift_cache.get(model_hires, shift_video, shift_audio)
            if model_hires is not None
            else None
        )
    else:
        shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
        model_use = _unpack(shifted)[0]
        if model_hires is not None:
            shifted_hires = MiniMaxH3SigmaShift.execute(
                model_hires, float(shift_video), float(shift_audio)
            )
            model_hires_use = _unpack(shifted_hires)[0]
        else:
            model_hires_use = None
    if callable(after_shift):
        remasked = after_shift(model_use, latent, sigmas)
        if remasked is not None:
            model_use = remasked
    sampler = _unpack(KSamplerSelect.execute("euler"))[0]
    if on_phase is not None:
        on_phase(phase_name, 0)
    try:
        output = sampler_cls().sample(
            model=model_use,
            positive=positive,
            negative=negative if negative else [],
            vae=vae,
            latent_image=latent,
            sampler=sampler,
            sigmas=sigmas,
            seed=int(seed),
            cfg=float(cfg),
            transition_step=int(_config_value(config, "selflift_transition_step", 5)),
            lowres_scale=float(_config_value(config, "selflift_lowres_scale", 0.4)),
            rho=float(_config_value(config, "selflift_rho", 0.6)),
            w_min=float(_config_value(config, "selflift_w_min", 0.5)),
            w_max=float(_config_value(config, "selflift_w_max", 1.0)),
            upscaler_model=upscaler_model,
            model_hires=model_hires_use,
            highres_tiling=bool(config.get("selflift_highres_tiling", False)),
        )
    finally:
        if callable(after_shift):
            try:
                from .h3_latent_continue import uninstall_continue_prefix_remask

                uninstall_continue_prefix_remask(model_use)
            except Exception:
                pass
        if shift_cache is None and model_use is not model:
            model_use = None
            model_hires_use = None
    if on_phase is not None:
        on_phase(phase_name, 1)
    return _unpack(output)[0]

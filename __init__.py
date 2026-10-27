"""ComfyUI nodes for the Anima reference adapter (inference).

The adapter reads a reference image as an extra latent frame:
  * the reference frame is fed clean (timestep 0) next to the noisy target frame;
  * its RoPE (h, w) coordinates are shifted ref_offset_h target heights down, so it is not
    copied pixel-aligned (no pose copying);
  * a LoRA, active only while the reference is attached, teaches the model to read it.

Nodes
  Anima Ref Adapter Loader   MODEL + adapter LoRA -> MODEL that understands the reference frame
  Anima Ref Latent           IMAGE -> letterboxed reference LATENT (+ empty target LATENT)
  Anima Ref Guider           positive / negative / reference with separate text and reference
                             guidance, plus APG or CFG-rescale correction
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.conds
import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
import comfy.samplers
import comfy.utils
import folder_paths
import node_helpers

REF_KEY = "anima_ref_latent"
REF_HALF_KEY = "anima_ref_latent_half"  # half-size encode for SPEED's low-resolution stage
WRAPPER_KEY = "anima_ref_adapter"
HOOKS_ATTR = "_anima_ref_adapter_hooks"  # on each hooked module: {adapter key: hook handle}
_adapters: dict[tuple, "_Adapter"] = {}  # adapter key -> loaded weights, shared by every loader run
_active = {"key": None}                  # key of the adapter whose LoRA the hooks apply right now
_warned_sizes: set = set()


def _adapter_key(path: str, strength: float) -> tuple:
    return os.path.abspath(path), os.path.getmtime(path), float(strength)


class _Adapter:
    """LoRA weights applied through forward hooks, only while this adapter's key is active.

    Hooks are tracked on the modules themselves and match by key, not by object identity, so a
    loader node that runs again (e.g. after the upstream model chain changed) keeps its LoRA working.
    """

    def __init__(self, path: str, strength: float):
        self.key = _adapter_key(path, strength)
        tensors = comfy.utils.load_torch_file(path, safe_load=True)
        self.weights = {}
        for key, down in tensors.items():
            if not key.endswith(".lora_down.weight"):
                continue
            name = key[len("diffusion_model."):-len(".lora_down.weight")]
            up = tensors[key.replace("lora_down", "lora_up")]
            alpha = float(tensors.get(key.replace("lora_down.weight", "alpha"), torch.tensor(down.shape[0])))
            self.weights[name] = [down, up, strength * alpha / down.shape[0]]
        if not self.weights:
            raise ValueError(f"{path} has no lora_down/lora_up weights")

    @classmethod
    def get(cls, path: str, strength: float) -> "_Adapter":
        key = _adapter_key(path, strength)
        if key not in _adapters:
            _adapters[key] = cls(path, strength)
        return _adapters[key]

    def install(self, diffusion_model) -> None:
        for name, entry in self.weights.items():
            module = diffusion_model.get_submodule(name)
            installed = getattr(module, HOOKS_ATTR, None)
            if installed is None:
                installed = {}
                setattr(module, HOOKS_ATTR, installed)
            if self.key in installed:
                continue

            def hook(module, inputs, output, entry=entry, key=self.key):
                if _active["key"] != key:
                    return output
                x = inputs[0]
                if entry[0].device != x.device or entry[0].dtype != x.dtype:
                    entry[0] = entry[0].to(device=x.device, dtype=x.dtype)
                    entry[1] = entry[1].to(device=x.device, dtype=x.dtype)
                return output + F.linear(F.linear(x, entry[0]), entry[1]) * entry[2]

            installed[self.key] = module.register_forward_hook(hook)


def _frame_positions(t_index, height, width, h0, w0, device):
    hh, ww = torch.meshgrid(torch.arange(height, device=device) + h0, torch.arange(width, device=device) + w0,
                            indexing="ij")
    return torch.full_like(hh, t_index).flatten(), hh.flatten(), ww.flatten()


def _rope_from_positions(pe, t, h, w):
    """VideoRopePosition3DEmb.generate_embeddings for explicit per-token (t, h, w) positions."""
    device = t.device
    h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(device))
    w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(device))
    t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(device))

    def rotations(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)

    emb = torch.cat([rotations(t, t_freqs), rotations(h, h_freqs), rotations(w, w_freqs)], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _pick_ref(refs, size):
    """The reference latent for a target latent of this (h, w): an exact-size encode if there is one
    (Anima Ref Latent also encodes a half-size reference for SPEED's low-resolution stage),
    else the closest one resized in latent space."""
    refs = [r.unsqueeze(2) if r.ndim == 4 else r for r in refs]
    for ref in refs:
        if tuple(ref.shape[-2:]) == size:
            return ref
    ref = min(refs, key=lambda r: abs(r.shape[-2] - size[0]) + abs(r.shape[-1] - size[1]))
    if size not in _warned_sizes:
        _warned_sizes.add(size)
        logging.warning(f"[Anima Ref Adapter] no reference encoded at latent size {size}; "
                        f"resizing the {tuple(ref.shape[-2:])} one (lower fidelity)")
    shrink = ref.shape[-2] > size[0]
    return F.interpolate(ref[:, :, 0].float(), size=size, mode="area" if shrink else "bilinear").unsqueeze(2)


def _ref_forward(dm, x, timesteps, context, refs, offset_ratio, transformer_options):
    """MiniTrainDIT._forward with the reference as a second frame (timestep 0, shifted RoPE)."""
    orig_shape = list(x.shape)
    patch = (dm.patch_temporal, dm.patch_spatial, dm.patch_spatial)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, patch)
    ref = _pick_ref(refs, tuple(x.shape[-2:])).to(device=x.device, dtype=x.dtype)
    frames = torch.cat([x[:, :, :1], ref[:, :, :1]], dim=2)
    x_emb, _, extra = dm.prepare_embedded_sequence(frames, fps=None, padding_mask=None)
    _, _, hp, wp, _ = x_emb.shape
    positions = [_frame_positions(0, hp, wp, 0, 0, x.device),
                 _frame_positions(1, hp, wp, round(offset_ratio * hp), 0, x.device)]
    rope = _rope_from_positions(dm.pos_embedder, *(torch.cat(axis) for axis in zip(*positions)))

    timesteps = timesteps.reshape(x.shape[0], -1)[:, :1]
    timesteps = torch.cat([timesteps, torch.zeros_like(timesteps)], dim=1)
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](timesteps).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    kwargs = {"rope_emb_L_1_1_D": rope.unsqueeze(1).unsqueeze(0), "adaln_lora_B_T_3D": adaln,
              "extra_per_block_pos_emb": extra, "transformer_options": transformer_options}
    if x_emb.dtype == torch.float16:
        x_emb = x_emb.float()  # residual stream in fp32, as MiniTrainDIT._forward does for fp16
    for index, block in enumerate(dm.blocks):
        transformer_options["block_index"] = index
        x_emb = block(x_emb, t_emb, context, **kwargs)
    out = dm.final_layer(x_emb.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln)
    out = dm.unpatchify(out)[:, :, :1]
    return out[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]


def _make_wrapper(adapter: _Adapter, offset_ratio: float):
    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        refs = [r for r in (kwargs.pop(REF_KEY, None), kwargs.pop(REF_HALF_KEY, None)) if r is not None]
        if not refs:
            return executor(x, timesteps, context, fps, padding_mask, **kwargs)
        adapter.install(executor.class_obj)  # no-op once hooked; covers a model reloaded since the loader ran
        previous = _active["key"]
        _active["key"] = adapter.key
        try:
            return _ref_forward(executor.class_obj, x, timesteps, context, refs, offset_ratio,
                                kwargs.get("transformer_options", {}))
        finally:
            _active["key"] = previous
    return wrapper


def _make_extra_conds(extra_conds, process_latent_in):
    # Object patches stay applied on the shared model, so a re-run loader may see its own earlier
    # patch here; wrap the original function, not the patch.
    extra_conds = getattr(extra_conds, "_anima_ref_inner", extra_conds)

    def patched(**kwargs):
        out = extra_conds(**kwargs)
        for key in (REF_KEY, REF_HALF_KEY):
            if kwargs.get(key) is not None:
                out[key] = comfy.conds.CONDRegular(process_latent_in(kwargs[key]))
        return out
    patched._anima_ref_inner = extra_conds
    return patched


class AnimaRefAdapterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "ref_offset_h": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                       "tooltip": "reference frame RoPE shift in target heights (training used 1.0)"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Anima Ref Adapter"

    def load(self, model, lora_name, strength, ref_offset_h):
        adapter = _Adapter.get(folder_paths.get_full_path_or_raise("loras", lora_name), strength)
        adapter.install(model.get_model_object("diffusion_model"))
        m = model.clone()
        m.add_object_patch("extra_conds", _make_extra_conds(m.get_model_object("extra_conds"),
                                                            m.get_model_object("process_latent_in")))
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                               _make_wrapper(adapter, ref_offset_h))
        return (m,)


class AnimaRefLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "vae": ("VAE",),
            "width": ("INT", {"default": 896, "min": 256, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 1152, "min": 256, "max": 2048, "step": 16}),
            "fill": ("INT", {"default": 255, "min": 0, "max": 255, "tooltip": "letterbox padding grey level"}),
        }}

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE")
    RETURN_NAMES = ("ref_latent", "empty_latent", "ref_image")
    FUNCTION = "encode"
    CATEGORY = "Anima Ref Adapter"

    @staticmethod
    def _letterbox(src, width, height, fill):
        scale = min(width / src.width, height / src.height)
        size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
        canvas = Image.new("RGB", (width, height), (fill, fill, fill))
        canvas.paste(src.resize(size, Image.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
        return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).unsqueeze(0)

    def encode(self, image, vae, width, height, fill):
        src = Image.fromarray((image[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
        pixels = self._letterbox(src, width, height, fill)
        ref = vae.encode(pixels)  # one image: the Wan VAE reads a batch as video frames
        # Half-size encode for samplers with a half-resolution stage (SPEED): shrinking the full latent
        # gives off-manifold latents that the adapter cannot read.
        half = vae.encode(self._letterbox(src, width // 16 * 8, height // 16 * 8, fill))
        empty = torch.zeros([1, 16, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return ({"samples": ref, "samples_half": half}, {"samples": empty}, pixels)


def _flat_dims(x):
    return tuple(range(1, x.ndim))


class AnimaRefGuiderImpl(comfy.samplers.CFGGuider):
    def set_settings(self, cfg, ref_guidance, correction, rescale_phi):
        self.cfg, self.ref_guidance, self.correction, self.rescale_phi = cfg, ref_guidance, correction, rescale_phi

    def set_ref_conds(self, positive, negative, ref_latent):
        refs = {REF_KEY: ref_latent["samples"]}
        if ref_latent.get("samples_half") is not None:
            refs[REF_HALF_KEY] = ref_latent["samples_half"]
        negative = node_helpers.conditioning_set_values(negative, {"prompt_type": "negative"})
        self.inner_set_conds({
            "positive": node_helpers.conditioning_set_values(positive, refs),
            "negative": node_helpers.conditioning_set_values(negative, refs),
            "negative_base": negative,
        })

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        conds = [self.conds.get("positive"), self.conds.get("negative")]
        if self.ref_guidance != 1.0:  # at 1 the reference is in both branches: plain CFG, no base branch
            conds.append(self.conds.get("negative_base"))
        out = comfy.samplers.calc_cond_batch(self.inner_model, conds, x, timestep, model_options)
        pos, neg = out[0], out[1]
        base = out[2] if len(out) > 2 else neg
        # x0 space: base + ref_guidance * (reference effect) + cfg * (text effect given the reference)
        guided = base + self.ref_guidance * (neg - base) + self.cfg * (pos - neg)
        if self.correction == "apg":
            update = (guided - pos).double()
            unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
            parallel = (update * unit).sum(dim=_flat_dims(pos), keepdim=True) * unit
            guided = pos + (update - parallel).to(pos.dtype)
        elif self.correction == "rescale":
            ratio = pos.std(dim=_flat_dims(pos), keepdim=True) / guided.std(dim=_flat_dims(guided), keepdim=True)
            guided = self.rescale_phi * guided * ratio + (1 - self.rescale_phi) * guided
        # Post-CFG model patches (e.g. Anima Safe PAG), called as comfy.samplers.cfg_function does.
        for fn in model_options.get("sampler_post_cfg_function", []):
            guided = fn({"denoised": guided, "cond": conds[0], "uncond": conds[1], "cond_scale": self.cfg,
                         "model": self.inner_model, "uncond_denoised": neg, "cond_denoised": pos, "sigma": timestep,
                         "model_options": model_options, "input": x})
        return guided


class AnimaRefGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "ref_latent": ("LATENT",),
            "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "text guidance"}),
            "ref_guidance": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1,
                                       "tooltip": "1 = reference in both CFG branches; higher pushes toward the reference"}),
            "correction": (["apg", "rescale", "none"],),
            "rescale_phi": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "Anima Ref Adapter"

    def get_guider(self, model, positive, negative, ref_latent, cfg, ref_guidance, correction, rescale_phi):
        guider = AnimaRefGuiderImpl(model)
        guider.set_ref_conds(positive, negative, ref_latent)
        guider.set_settings(cfg, ref_guidance, correction, rescale_phi)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "AnimaRefAdapterLoader": AnimaRefAdapterLoader,
    "AnimaRefLatent": AnimaRefLatent,
    "AnimaRefGuider": AnimaRefGuider,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaRefAdapterLoader": "Anima Ref Adapter Loader",
    "AnimaRefLatent": "Anima Ref Latent",
    "AnimaRefGuider": "Anima Ref Guider",
}

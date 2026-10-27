# ComfyUI-Anima-Ref-Adapter

Reference-image character and style transfer for [Anima](https://huggingface.co/circlestone-labs/Anima)
in ComfyUI. Give one reference image and a prompt that describes only the scene; the adapter keeps the
character (hair, eyes, outfit, accessories) and some of the drawing style, while pose, framing and
background follow the prompt.

- Weights: [ElectronGirl/anima-ref-adapter](https://huggingface.co/ElectronGirl/anima-ref-adapter) (LoRA, CircleStone Non-Commercial License)
- Method and results: [docs/report.md](docs/report.md)

![held-out characters](docs/figures/fig2_heldout.jpg)

*Reference · adapter with a scene-only prompt · same prompt without adapter · image generated with full
character tags. These characters were not in the training set.*

> **Pilot release.** Trained on 3,600 synthetic images of 300 characters for 2,000 steps. Identity
> transfer works well; fine details (small patterns, scars, skin tones) and painterly styles are weak.

## How it works

The reference is appended to the latent as a second video frame, as in ComfyUI-Cosmos-Reference, but
it is fed clean (timestep 0) and its RoPE coordinates are shifted past the target grid, so the model
cannot copy it pixel-aligned (no pose copying). A rank-32 LoRA, active only while the reference is
attached, learns to read it. See the [report](docs/report.md).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ElectronGirl/ComfyUI-Anima-Ref-Adapter.git
```

Download `anima_ref_adapter_pilot_r32.safetensors` from [ElectronGirl/anima-ref-adapter](https://huggingface.co/ElectronGirl/anima-ref-adapter)
into `ComfyUI/models/loras/`. You also need Anima base v1.0 (`anima_baseV10.safetensors` /
`anima-base-v1.0.safetensors`), the Qwen3-0.6B text encoder and the Qwen-Image VAE from
[circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima).

Restart ComfyUI and reload the browser tab (Ctrl+F5).

## Nodes

| Node | What it does |
|---|---|
| **Anima Ref Adapter Loader** | Attaches the adapter to an Anima MODEL. `strength` scales the LoRA; keep `ref_offset_h` at 1.0 (training value). |
| **Anima Ref Latent** | Letterboxes the reference image into the output size (white padding) and VAE-encodes it, plus a half-size encode for samplers with a half-resolution stage. Also returns the empty latent for that size. |
| **Anima Ref Guider** | GUIDER for SamplerCustomAdvanced with separate text guidance (`cfg`) and reference guidance (`ref_guidance`), and an `apg` / `rescale` correction against over-saturation. Runs post-CFG model patches such as PAG. |

## Workflows

- [`workflows/anima_ref_adapter.json`](workflows/anima_ref_adapter.json): basic (Euler, 30 steps).
- [`workflows/anima_ref_adapter_pag.json`](workflows/anima_ref_adapter_pag.json): adds
  Anima Safe PAG (`comfyui-anima-safe-pag`) on block 18; needs that node pack and ComfyUI-KJNodes
  (SageAttention).

Recommended settings:

- Prompt: quality tags, gender, `solo`, scene tags (framing, pose, expression, background, lighting).
  Leave out the character name, artist tags and appearance tags.
- `cfg` 4.5, `ref_guidance` 4.5–6, `correction` = `apg`. `ref_guidance` 1 is faster (no extra base
  branch) but much weaker.
- Base model without the turbo LoRA; bf16 compute (`ModelComputeDtype`), since Anima overflows in fp16.
- Single-resolution samplers (Euler, ER-SDE). SamplerSPEED's half-resolution stage is unreliable with
  this adapter.

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). The adapter weights are a derivative of the CircleStone Model
and are distributed separately under the CircleStone Non-Commercial License (see the Hugging Face
repository). This project is not affiliated with or endorsed by CircleStone Labs.

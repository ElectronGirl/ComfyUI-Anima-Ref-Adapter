# Anima Reference Adapter 사용 가이드

[English](guide.en.md) · **한국어** · [日本語](guide.ja.md)

이 가이드는 ComfyUI에 어댑터를 설치하고, 첫 이미지를 만들고, 프롬프트를 쓰고, 두 캐릭터를 함께 그리고,
설정을 조절하고, 자주 생기는 문제를 해결하는 방법을 순서대로 설명합니다. 작동 원리와 측정 결과는
[README](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/blob/main/README.ko.md)에 있습니다.

## 1. 준비물

- Anima를 돌릴 수 있는 **ComfyUI**. 이미 Anima로 이미지를 만들고 있다면 준비가 끝난 것입니다.
- [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)에서 받는 Anima 파일 3개:
  - `anima-base-v1.0.safetensors` → `ComfyUI/models/diffusion_models/`
  - `qwen_3_06b_base.safetensors` → `ComfyUI/models/text_encoders/`
  - `qwen_image_vae.safetensors` → `ComfyUI/models/vae/`
- Anima를 돌리는 데 필요한 VRAM 외에 약 1GB가 더 필요합니다. 1MP 레퍼런스의 key/value를 캐시해 두는 데 쓰입니다.
  8GB인 RTX 3060 Ti에서 테스트했습니다.

어댑터 가중치는 따로 받지 않아도 됩니다. 처음 실행할 때 노드가 직접 받아옵니다.

## 2. 노드 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter.git
```

ComfyUI를 재시작합니다. git이 없다면 GitHub에서 ZIP으로 받아(Code → Download ZIP)
`ComfyUI/custom_nodes/`에 압축을 풀면 됩니다.

## 3. 첫 이미지 만들기

1. [`anima_ref_adapter.json`](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/blob/main/workflows/anima_ref_adapter.json)을
   받아서 ComfyUI 화면에 끌어다 놓습니다. Anima의 기본 text-to-image 워크플로에 **Anima Ref Adapter** 노드
   하나를 모델 로더와 KSampler 사이에 더한 것입니다.
2. **Load Image** 노드에서 레퍼런스 이미지를 고릅니다.
3. 긍정 프롬프트에는 퀄리티 태그를 그대로 두고 장면만 씁니다(4장 참고).
4. 실행합니다. 처음 실행할 때 `anima_ref_adapter_v2_r64.safetensors`(183MB)를 `ComfyUI/models/loras/`로
   받아오고, 그다음부터는 바로 시작합니다.

<img src="https://raw.githubusercontent.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/main/docs/showcase/v2_police.jpg" width="100%" alt="레퍼런스와 결과">

## 4. 프롬프트 쓰는 법

캐릭터(얼굴, 머리, 눈, 의상, 장신구)와 그림체의 상당 부분은 레퍼런스에서 가져옵니다. 나머지는 모두
프롬프트에서 정합니다.

| 쓸 것 | 쓰지 않을 것 |
|---|---|
| 퀄리티 태그: `masterpiece, best quality, score_7, safe,` | 캐릭터 이름 |
| `1girl` 또는 `1boy`, 그리고 `solo` | 작가 태그 |
| 구도: `cowboy shot`, `full body`, `upper body`, `portrait` | 머리·눈·의상 태그(몇 개 정도는 괜찮습니다) |
| 자세와 동작: `sitting`, `lying, on side`, `holding cup` | |
| 표정: `smile`, `sleepy`, `laughing` | |
| 장소·시간·조명: `cafe, indoors, day`, `beach, sunset` | |

예시: `masterpiece, best quality, score_7, safe, 1girl, solo, cowboy shot, sitting, cafe, holding cup, smile, looking at viewer, indoors, window, day`

도움이 되는 점:

- 레퍼런스와 다른 자세를 쓰는 것이 좋습니다. 어댑터는 레퍼런스의 자세를 따라 하지 않도록 학습되었고,
  프롬프트에 자세가 분명하게 적혀 있으면 그 효과가 더 확실해집니다.
- 레퍼런스에 보이지 않는 부분(상반신 이미지의 다리, 의상의 뒷면 등)은 모델이 추측해서 그립니다.
  캐릭터 전체가 보이는 레퍼런스를 쓰면 의상이 더 일정하게 유지됩니다.
- 부정 프롬프트는 Anima 모델 카드에 있는 것을 그대로 쓰면 됩니다.

## 5. 두 캐릭터 함께 그리기

<img src="https://raw.githubusercontent.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/main/docs/showcase/v2_duo_police_mecha.jpg" width="100%" alt="레퍼런스 두 장과 결과">

1. [`anima_ref_adapter_two_characters.json`](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/blob/main/workflows/anima_ref_adapter_two_characters.json)을
   엽니다. **Load Image** 노드 두 개를 기본 **Image Stitch** 노드가 좌우로 이어 붙입니다.
2. 첫 번째 이미지에는 왼쪽에 올 캐릭터를, 두 번째 이미지에는 오른쪽에 올 캐릭터를 넣습니다.
3. `solo` 대신 `2girls`(또는 `1boy, 1girl`, `2boys`)를 쓰고, 누가 어느 쪽인지 몇 단어로 적습니다.
   `2girls, the girl on the left is a policewoman with black hair over one eye, the girl on the right is a mecha girl with aqua hair and mechanical wings, standing side by side, peace sign, smile, city, street, day`
4. 1216x832처럼 가로가 긴 크기를 씁니다.

두 캐릭터는 섞이지 않고 따로 그려지지만, 그림체는 둘의 중간쯤으로 하나로 맞춰지는 경향이 있습니다.

## 6. 설정

모두 **Anima Ref Adapter** 노드에 있습니다.

| 설정 | 기본값 | 역할 |
|---|---|---|
| `ref_guidance` | 4.5 | 레퍼런스 쪽으로 미는 세기입니다. 캐릭터가 달라지면 5~6으로 올리고, 색이 너무 진해지면 낮춥니다. 1로 두면 모델을 한 번 덜 호출하므로 빠르지만 약해집니다. |
| `correction` | `apg` | 가이던스가 높을 때 색이 과해지는 것을 막습니다. `rescale`도 쓸 수 있고, `none`은 끕니다. |
| `ref_end` | 1.0 | 샘플링 진행 중 이 지점부터 `ref_guidance`를 멈춥니다. 레퍼런스 자체는 끝까지 유지됩니다. 0.6 정도로 두면 거의 같은 그림을 더 빨리 얻습니다. |
| `ref_megapixels` | 1.0 | 레퍼런스를 줄이거나 늘리는 크기입니다. VRAM이 부족하면 0.5~0.7로 낮춥니다. |
| `strength` | 1.0 | 어댑터 LoRA의 세기입니다. |

다른 LoRA(그림체, 디테일 등)는 어댑터 노드 **앞**, 즉 모델 로더와 Anima Ref Adapter 사이에 둡니다.

## 7. 더 빠르게

| 방법 | 설정 | 효과 (RTX 3060 Ti, 832x1216) |
|---|---|---|
| turbo LoRA | Anima base용 turbo LoRA를 어댑터 앞에 두고 10 steps, cfg 1, `ref_guidance` 1 | 67초 → 11초. 레퍼런스 그림체는 약해집니다. |
| `ref_guidance` 1 | 노드에서 설정 | 67초 → 50초 |
| SageAttention | KJNodes의 **Patch Sage Attention**을 어댑터 앞에 둠 | 약 7% |
| EasyCache | 기본 **EasyCache** 노드를 어댑터 뒤에 둠 | 약 7% |
| `ref_end` 0.6 | 노드에서 설정 | 약 7% |

## 8. 문제 해결

| 증상 | 해결 방법 |
|---|---|
| 레퍼런스가 아니라 평범한 Anima 그림이 나옵니다 | KSampler의 `model` 입력이 모델 로더가 아니라 **Anima Ref Adapter**에서 오는지 확인합니다. `ref_guidance`를 올려 봅니다. |
| 색이 너무 진하거나 탄 것처럼 보입니다 | `ref_guidance`를 3~4로 낮추고 `correction`은 `apg`로 둡니다. |
| 레퍼런스의 자세를 그대로 따라 합니다 | 프롬프트에 레퍼런스와 다른 자세와 구도를 분명하게 씁니다. |
| 레퍼런스 캐릭터가 장면 안에 포스터나 피규어로 다시 나옵니다 | 알려진 한계입니다. 시드를 바꾸거나 `ref_guidance`를 낮춥니다. |
| 자동 다운로드가 실패합니다 | [`anima_ref_adapter_v2_r64.safetensors`](https://huggingface.co/ElectronGirl/anima-ref-adapter/blob/main/anima_ref_adapter_v2_r64.safetensors)를 직접 받아 `ComfyUI/models/loras/`에 넣습니다. |
| 메모리가 부족합니다 | `ref_megapixels`를 낮추거나 출력 크기를 줄입니다. |
| 파일럿 어댑터를 쓰고 싶습니다 | [`anima_ref_adapter_advanced.json`](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/blob/main/workflows/anima_ref_adapter_advanced.json)을 씁니다. 파일럿은 레퍼런스가 출력과 같은 크기여야 합니다. |

## 9. Forge Neo

Stable Diffusion WebUI Forge (Neo)용 확장도 있습니다:
[sd-forge-anima-ref-adapter](https://github.com/ElectronGirl/sd-forge-anima-ref-adapter).

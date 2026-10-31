<div align="center">

<img src="docs/showcase/banner.jpg" width="100%" alt="Anima Reference Adapter v2">

# Anima Reference Adapter for ComfyUI

**레퍼런스 이미지에서 캐릭터와 그림체를, 프롬프트에서 자세·장소·구도를.**

[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-anima--ref--adapter-ffcc4d)](https://huggingface.co/ElectronGirl/anima-ref-adapter)
[![Nodes](https://img.shields.io/badge/%F0%9F%A7%A9%20ComfyUI-nodes-1f6feb)](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter)
[![Training code](https://img.shields.io/badge/%F0%9F%94%AC%20training-code-8a63d2)](https://github.com/ElectronGirl/anima-ref-adapter-research)

[쇼케이스](#쇼케이스) · [빠른 시작](#빠른-시작) · [사용 가이드](docs/guide.ko.md) · [속도](#속도와-호환성) · [학습 코드](https://github.com/ElectronGirl/anima-ref-adapter-research)

[English](README.md) · **한국어** · [日本語](README.ja.md)

</div>

<br>

[Anima](https://huggingface.co/circlestone-labs/Anima)에 레퍼런스 이미지 한 장과 장면만 적은 프롬프트를
주면, 어댑터가 **누구를** 그리는지(머리, 눈, 의상, 장신구)와 **어떻게** 그리는지의 상당 부분을 레퍼런스에서
가져옵니다. **자세, 배경, 구도는 프롬프트를 따릅니다.** 캐릭터 이름이나 작가·의상 태그가 필요 없으므로
오리지널 캐릭터에도 쓸 수 있습니다. 레퍼런스 두 장을 나란히 두면 두 캐릭터를 한 장에 그릴 수 있습니다.

> [!NOTE]
> **자금 부족으로 개발이 중단되었습니다.** 지금은 추가 학습을 할 수 없고, 계획했던 캐릭터·그림체 분리
> 레퍼런스는 공개하지 못했습니다. 학습 코드와 데이터 생성 코드, 실험 기록은
> **[anima-ref-adapter-research](https://github.com/ElectronGirl/anima-ref-adapter-research)**에 공개되어 있습니다.

## v2의 새로운 점

| | |
|---|---|
| ⚡ **파일럿보다 약 2배 빠름** | RTX PRO 6000에서 샘플링 스텝당 2.1배 빠릅니다. RTX 3060 Ti에서는 832x1216 이미지 한 장이 105초에서 65초로 줄었고, 레퍼런스 때문에 늘어나는 시간은 72초에서 32초로 줄었습니다. 레퍼런스의 KV를 스텝마다 다시 계산하지 않고 이미지당 한 번만 계산해 캐시합니다. |
| 🧩 **노드 하나** | **Anima Ref Adapter** 노드 하나를 Anima 기본 워크플로에 끼우면 KSampler를 비롯한 어떤 샘플러와도 동작합니다. 가중치는 처음 실행할 때 자동으로 받아옵니다. |
| 👥 **두 캐릭터** | 레퍼런스 두 장을 나란히 두고, 프롬프트에 누가 어느 쪽인지 적으면 됩니다. |
| 📚 **더 많은 학습 데이터** | 캐릭터 2,300명, 작가 그림체 1,952개, 이미지 15,600장으로 학습했습니다(파일럿: 이미지 3,600장, 캐릭터 300명). rank 64입니다. |

## 쇼케이스

[노드 하나짜리 워크플로](workflows/anima_ref_adapter.json)를 Anima 기본 설정으로 돌린 결과입니다: er_sde, 30 steps,
cfg 4.5, `ref_guidance` 4.5 + APG, 프롬프트는 `masterpiece, best quality, score_7, safe, 1girl, solo, <장면>`, 부정 프롬프트는 Anima 모델 카드의 것을
썼습니다. 다른 LoRA는 쓰지 않았습니다. 이미지마다 시드 하나만 썼고 골라내지 않았습니다. 단, 누운 자세는 6장 중
가장 나은 것을 골랐습니다.

### 레퍼런스 한 장

<img src="docs/showcase/v2_police.jpg" width="100%" alt="reference and two scenes">
<sub><code>cowboy shot, sitting, cafe, holding cup, coffee, smile, looking at viewer, indoors, window, day</code> &nbsp;·&nbsp; <code>full body, lying, on side, on bed, head rest, looking at viewer, smile, bedroom, indoors, night, lamp light</code></sub>

<img src="docs/showcase/v2_mecha.jpg" width="100%" alt="reference and two scenes">
<sub><code>full body, flying, from below, sky, clouds, sunlight, dynamic pose, outstretched arm</code> &nbsp;·&nbsp; <code>sitting, on floor, hugging own legs, sleepy, hangar, indoors, dim lighting</code></sub>

<img src="docs/showcase/v2_swim.jpg" width="100%" alt="reference and two scenes">
<sub><code>cowboy shot, beach, ocean, sunset, wind, hand on own hat, sun hat, smile, looking back</code> &nbsp;·&nbsp; <code>upper body, sitting, poolside, legs in water, splashing, laughing, day</code></sub>

### 두 캐릭터

레퍼런스 두 장을 한 이미지에 나란히 붙이고(기본 **Image Stitch** 노드), 왼쪽과 오른쪽이 각각 누구인지 몇 단어로
적습니다. `solo` 대신 `2girls`나 `1boy, 1girl`을 쓰고, 가로가 긴 크기를 씁니다. [워크플로](workflows/anima_ref_adapter_two_characters.json).

<img src="docs/showcase/v2_duo_police_mecha.jpg" width="100%" alt="two references and the result">
<sub><code>2girls, the girl on the left is a policewoman with black hair over one eye, the girl on the right is a mecha girl with aqua hair and mechanical wings, standing side by side, peace sign, smile, city, street, day</code></sub>

<img src="docs/showcase/v2_duo_police_swim.jpg" width="100%" alt="two references and the result">
<sub><code>2girls, the girl on the left is a policewoman with black hair over one eye, the girl on the right is a girl with short black hair in a blue frilled bikini, sitting together on a bench, eating ice cream, park, day</code></sub>

<img src="docs/showcase/v2_duo_mecha_swim.jpg" width="100%" alt="two references and the result">
<sub><code>2girls, the girl on the left is a mecha girl with aqua hair and mechanical wings, the girl on the right is a girl with short black hair in a blue frilled bikini, high five, beach, ocean, day</code></sub>

## 빠른 시작

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter.git
```

1. Anima base v1.0, Qwen3-0.6B 텍스트 인코더, Qwen-Image VAE를
   [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)에서 받습니다. Anima를 쓸 때와 같습니다.
2. ComfyUI를 재시작하고 [`workflows/anima_ref_adapter.json`](workflows/anima_ref_adapter.json)을 엽니다. Anima
   text-to-image 워크플로에 노드 하나를 더한 것이며, 다른 커스텀 노드는 필요 없습니다.
3. 레퍼런스 이미지를 넣고, 장면만 적은 프롬프트를 쓰고, 실행합니다. 처음 실행할 때 `anima_ref_adapter_v2_r64.safetensors`(183MB)를
   [Hugging Face](https://huggingface.co/ElectronGirl/anima-ref-adapter)에서 `ComfyUI/models/loras/`로 받아옵니다. 직접 설치하려면
   [이 파일](https://huggingface.co/ElectronGirl/anima-ref-adapter/blob/main/anima_ref_adapter_v2_r64.safetensors)을 같은 폴더에 넣으면 됩니다.

**직접 만든 워크플로에서는** **Anima Ref Adapter**를 모델(다른 LoRA 뒤)과 샘플러 사이에 두고, VAE와 레퍼런스
이미지를 연결합니다.

프롬프트 쓰는 법, 설정, 문제 해결까지 단계별로 정리한 **[사용 가이드](docs/guide.ko.md)**가 있습니다.

## 속도와 호환성

RTX 3060 Ti 8GB(fp16), 832x1216, 레퍼런스 한 장, 두 번 잰 것 중 빠른 쪽:

| 설정 | 이미지 한 장 |
|---|---|
| Anima base, er_sde 30 steps, cfg 4.5, **레퍼런스 없음** | 33 초 |
| + 파일럿 어댑터, `ref_guidance` 4.5 | 105 초 |
| **+ v2 어댑터**, `ref_guidance` 4.5 | **65-67** 초 |
| `ref_guidance` 1 (추가 모델 호출 없음) | 50 초 |
| + SageAttention ([KJNodes](https://github.com/kijai/ComfyUI-KJNodes) Patch Sage Attention) | 62 초 |
| + EasyCache (기본 노드) | 63 초 |
| `ref_end` 0.6 | 62 초 |
| **turbo LoRA**(strength 1), 10 steps, cfg 1, `ref_guidance` 1 | **11** 초 |

<img src="docs/figures/v2_compat.jpg" width="100%" alt="speed settings with the same seed">

- SageAttention, EasyCache, `ref_end` 0.6은 이 GPU에서 각각 약 7%씩 빨라지고, 그림은 거의 같습니다.
- Anima base용 **turbo LoRA**(테스트한 파일: `anima-turbo-lora-v0.2`)는 어댑터 앞에 둡니다. 캐릭터는 유지되지만
  그림이 turbo 특유의 화풍 쪽으로 끌려가서, 레퍼런스의 그림체는 덜 반영됩니다.
- 테스트하지 않은 것: Anima-Turbo·Anima-Aesthetic 전체 모델, SamplerSPEED, torch.compile.

<details>
<summary>turbo LoRA로 만든 쇼케이스 전체 (10 steps, cfg 1)</summary>

<img src="docs/figures/v2_turbo.jpg" width="100%" alt="turbo LoRA">

</details>

## 권장 설정

|  | 권장값 | 설명 |
|---|---|---|
| 기본 모델 | Anima base v1.0 | 어댑터를 이 모델로 학습했습니다. turbo LoRA도 동작합니다. |
| 샘플러 | 아무거나; er_sde나 Euler, 30 steps, `simple` | 어댑터 없이 Anima를 쓸 때와 같습니다. |
| `cfg` | 4.5 | 텍스트 가이던스입니다. turbo LoRA를 쓸 때는 1입니다. |
| `ref_guidance` | 4.5 | 1이면 빠르지만 약하고, 높을수록 캐릭터 세부와 그림체가 강해지며 채도도 올라갑니다. |
| `correction` | `apg` | 또는 `rescale` |
| `ref_end` | 1.0 | 0.6 정도면 거의 같은 그림을 더 빨리 얻습니다. |
| 프롬프트 | 퀄리티 + `1girl`/`1boy` + `solo` + 장면 태그 | 캐릭터 이름, 작가 태그, 외모 태그는 넣지 않습니다. 외모 태그 몇 개는 괜찮습니다. |
| 다른 LoRA | 어댑터 노드 앞 |  |

## 작동 원리

레퍼런스 토큰은 자기 자신만 보고 텍스트 cross-attention을 건너뛰도록, 생성 중인 이미지의 토큰은 자기 자신과
레퍼런스를 함께 보도록 학습했습니다(비대칭 attention). 그래서 레퍼런스 쪽 계산은 스텝·프롬프트·CFG 분기와
무관하고, 블록별 key/value를 이미지당 한 번만 계산해 재사용합니다. 레퍼런스는 생성 이미지 격자 바깥으로 밀린
RoPE 좌표에 놓이기 때문에, 자세를 그대로 베끼지 않고 외형으로 읽힙니다. 자세한 내용은
[English README](README.md#how-it-works)와 [기술 보고서](docs/report.md)(파일럿)를 참고해 주세요.

## 파일럿 (v1)

파일럿(`anima_ref_adapter_pilot_r32.safetensors`, rank 32)도 계속 받을 수 있고,
[advanced 워크플로](workflows/anima_ref_adapter_advanced.json)로 쓸 수 있습니다. 파일럿 갤러리와 학습에서 뺀 캐릭터의
결과는 [English README](README.md#pilot-v1)에 있습니다.

## 한계

- 캐릭터는 그림체보다 훨씬 잘 옮겨집니다. 붓질이 강한 그림체는 거의 옮겨지지 않습니다.
- 작은 무늬, 흉터, 귀걸이, 특이한 피부색은 자주 빠지고, 큰 의상은 색이 달라질 수 있습니다.
- 레퍼런스 캐릭터가 장면 안에 사물(포스터 등)로 다시 나타날 때가 있습니다.
- Anima가 생성한 이미지로만 학습했으므로 기본 모델의 편향을 그대로 가집니다.

## 로드맵

| | |
|---|---|
| 레퍼런스 한 장에서 캐릭터와 그림체 | ✅ 파일럿, v2 |
| 레퍼런스 캐시를 쓰는 비대칭 attention | ✅ v2 |
| 레퍼런스 두 장을 나란히 두어 두 캐릭터 | ✅ v2 |
| 캐릭터·그림체 분리 레퍼런스 | ⏸️ 중단 (실험만 진행) |
| 고해상도 학습, 붓질이 강한 그림체와 세부 개선 | ⏸️ 중단 |

## 라이선스

노드 코드는 **Apache-2.0**([LICENSE](LICENSE))입니다. 어댑터 가중치는 CircleStone Model의 파생물이며 Hugging
Face에서 **CircleStone Non-Commercial License**로 배포합니다. 이 프로젝트는 CircleStone Labs와 관계가 없고
CircleStone Labs의 보증을 받지 않았습니다.

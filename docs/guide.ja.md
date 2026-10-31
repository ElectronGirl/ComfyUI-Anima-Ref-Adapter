# Anima Reference Adapter 使い方ガイド

[English](guide.en.md) · [한국어](guide.ko.md) · **日本語**

このガイドでは、ComfyUI へのアダプターのインストール、最初の画像の生成、プロンプトの書き方、2人のキャラクターを
一緒に描く方法、設定の調整、よくある問題の解決方法を順に説明します。仕組みと測定結果は
[README](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/blob/main/README.ja.md) をご覧ください。

## 1. 必要なもの

- Anima が動く **ComfyUI**。すでに Anima で画像を生成できていれば、準備は完了しています。
- [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) から入手する Anima のファイル3つ:
  - `anima-base-v1.0.safetensors` → `ComfyUI/models/diffusion_models/`
  - `qwen_3_06b_base.safetensors` → `ComfyUI/models/text_encoders/`
  - `qwen_image_vae.safetensors` → `ComfyUI/models/vae/`
- Anima に必要な VRAM に加えて約 1GB。1MP の参照画像の key/value をキャッシュするために使います。
  8GB の RTX 3060 Ti で動作を確認しています。

アダプターの重みを別途ダウンロードする必要はありません。初回実行時にノードが自動で取得します。

## 2. ノードのインストール

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter.git
```

ComfyUI を再起動してください。git がない場合は、GitHub から ZIP をダウンロードし（Code → Download ZIP）、
`ComfyUI/custom_nodes/` に展開します。

## 3. 最初の画像

1. [`anima_ref_adapter.json`](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/blob/main/workflows/anima_ref_adapter.json)
   をダウンロードし、ComfyUI の画面にドラッグします。Anima 標準の text-to-image ワークフローに、
   **Anima Ref Adapter** ノードを1つ（モデルローダーと KSampler の間に）加えたものです。
2. **Load Image** ノードで参照画像を選びます。
3. ポジティブプロンプトは品質タグをそのまま残し、シーンだけを書きます（4章を参照）。
4. 実行します。初回は `anima_ref_adapter_v2_r64.safetensors`（183MB）を `ComfyUI/models/loras/` に
   ダウンロードし、2回目以降はすぐに始まります。

<img src="https://raw.githubusercontent.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/main/docs/showcase/v2_police.jpg" width="100%" alt="参照画像と結果">

## 4. プロンプトの書き方

キャラクター（顔、髪、目、衣装、アクセサリー）と画風の多くは参照画像から取り込みます。それ以外はすべて
プロンプトで決めます。

| 書くもの | 書かないもの |
|---|---|
| 品質タグ: `masterpiece, best quality, score_7, safe,` | キャラクター名 |
| `1girl` または `1boy`、そして `solo` | 絵師タグ |
| 構図: `cowboy shot`, `full body`, `upper body`, `portrait` | 髪・目・衣装のタグ（少しなら問題ありません） |
| ポーズと動作: `sitting`, `lying, on side`, `holding cup` | |
| 表情: `smile`, `sleepy`, `laughing` | |
| 場所・時間・光: `cafe, indoors, day`, `beach, sunset` | |

例: `masterpiece, best quality, score_7, safe, 1girl, solo, cowboy shot, sitting, cafe, holding cup, smile, looking at viewer, indoors, window, day`

ポイント:

- 参照画像とは違うポーズを書くのがおすすめです。アダプターは参照画像のポーズをまねしないように学習されて
  おり、プロンプトにポーズがはっきり書かれていると、その効果がより確実になります。
- 参照画像に写っていない部分（バストアップ画像の脚、衣装の背面など）はモデルが推測して描きます。
  キャラクター全体が写った参照画像を使うと、衣装がより安定します。
- ネガティブプロンプトは Anima のモデルカードにあるものをそのまま使えます。

## 5. 2人のキャラクター

<img src="https://raw.githubusercontent.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/main/docs/showcase/v2_duo_police_mecha.jpg" width="100%" alt="参照画像2枚と結果">

1. [`anima_ref_adapter_two_characters.json`](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/blob/main/workflows/anima_ref_adapter_two_characters.json)
   を開きます。2つの **Load Image** ノードを、標準の **Image Stitch** ノードが左右につなぎます。
2. 1枚目に左のキャラクター、2枚目に右のキャラクターを読み込みます。
3. `solo` の代わりに `2girls`（または `1boy, 1girl`、`2boys`）と書き、どちらが誰かを短く書きます:
   `2girls, the girl on the left is a policewoman with black hair over one eye, the girl on the right is a mecha girl with aqua hair and mechanical wings, standing side by side, peace sign, smile, city, street, day`
4. 1216x832 のような横長のサイズを使います。

2人のキャラクターは混ざらずに描き分けられますが、画風は2人の中間あたりで1つにまとまる傾向があります。

## 6. 設定

すべて **Anima Ref Adapter** ノードにあります。

| 設定 | 既定値 | 役割 |
|---|---|---|
| `ref_guidance` | 4.5 | 参照画像に寄せる強さです。キャラクターがずれるときは 5〜6 に上げ、色が濃すぎるときは下げます。1 にするとモデルの呼び出しが1回減るので速くなりますが、効果は弱くなります。 |
| `correction` | `apg` | ガイダンスが強いときの色の行き過ぎを抑えます。`rescale` も使え、`none` で無効になります。 |
| `ref_end` | 1.0 | サンプリングのこの時点以降は `ref_guidance` を止めます（参照画像そのものは最後まで残ります）。0.6 前後にすると、ほぼ同じ画像をより速く得られます。 |
| `ref_megapixels` | 1.0 | 参照画像をこの面積に縮小・拡大します。VRAM が足りないときは 0.5〜0.7 に下げます。 |
| `strength` | 1.0 | アダプター LoRA の強さです。 |

ほかの LoRA（画風、ディテールなど）はアダプターノードの**前**、つまりモデルローダーと Anima Ref Adapter の
間に置きます。

## 7. 高速化

| 方法 | 設定 | 効果（RTX 3060 Ti、832x1216） |
|---|---|---|
| turbo LoRA | Anima base 用の turbo LoRA をアダプターの前に置き、10 steps、cfg 1、`ref_guidance` 1 | 67秒 → 11秒。参照画像の画風は弱くなります。 |
| `ref_guidance` 1 | ノードで設定 | 67秒 → 50秒 |
| SageAttention | KJNodes の **Patch Sage Attention** をアダプターの前に置く | 約7% |
| EasyCache | 標準の **EasyCache** ノードをアダプターの後に置く | 約7% |
| `ref_end` 0.6 | ノードで設定 | 約7% |

## 8. トラブルシューティング

| 症状 | 対処 |
|---|---|
| 参照画像ではなく、普通の Anima の画像になる | KSampler の `model` がモデルローダーから直接ではなく、**Anima Ref Adapter** から来ているか確認してください。`ref_guidance` を上げてみてください。 |
| 色が濃すぎる、焼けたように見える | `ref_guidance` を 3〜4 に下げ、`correction` は `apg` のままにします。 |
| 参照画像のポーズをそのまま写してしまう | プロンプトに、参照画像とは違うポーズと構図をはっきり書きます。 |
| 参照キャラクターがシーンの中にポスターやフィギュアとして再登場する | 既知の制限です。シードを変えるか、`ref_guidance` を下げてください。 |
| 自動ダウンロードに失敗する | [`anima_ref_adapter_v2_r64.safetensors`](https://huggingface.co/ElectronGirl/anima-ref-adapter/blob/main/anima_ref_adapter_v2_r64.safetensors) を手動でダウンロードし、`ComfyUI/models/loras/` に置きます。 |
| メモリが足りない | `ref_megapixels` を下げるか、出力サイズを小さくします。 |
| パイロット版アダプターを使いたい | [`anima_ref_adapter_advanced.json`](https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter/blob/main/workflows/anima_ref_adapter_advanced.json) を使ってください。パイロット版は参照画像が出力と同じサイズである必要があります。 |

## 9. Forge Neo

Stable Diffusion WebUI Forge (Neo) 用の拡張機能もあります:
[sd-forge-anima-ref-adapter](https://github.com/ElectronGirl/sd-forge-anima-ref-adapter)。

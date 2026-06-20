# Qwen3.5-0.8B 日本語 QLoRA 微調整

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Unsloth](https://img.shields.io/badge/Unsloth-supported-green.svg)](https://github.com/unslothai/unsloth)
[![Colab](https://img.shields.io/badge/Google%20Colab-T4-orange.svg)](https://colab.research.google.com/)

> Qwen3.5-0.8B (FP16) を Unsloth + QLoRA で日本語強化学習する Google Colab ノートブック。
> `llm-jp/oasst1-21k-ja` から抽出した高品質 8,000 件を用い、**破滅的忘却を最小限に抑えながら** 日本語性能を底上げすることを目的とする。

## 目次

- [概要](#概要)
- [クイックスタート](#クイックスタート)
- [アーキテクチャと設計思想](#アーキテクチャと設計思想)
- [ハイパーパラメータ](#ハイパーパラメータ)
- [データセット](#データセット)
- [ノートブックの構成](#ノートブックの構成)
- [Qwen3.5 特有の注意点](#qwen35-特有の注意点)
- [トラブルシューティング](#トラブルシューティング)
- [評価と次のステップ](#評価と次のステップ)
- [ファイル構成](#ファイル構成)
- [ライセンス](#ライセンス)
- [参考資料](#参考資料)

---

## 概要

本リポジトリは、Qwen3.5-0.8B の日本語性能を向上させるための学習パイプラインを提供します。
「軽量モデル×日本語特化」というニーズに対し、以下の設計方針で実装しています:

1. **破滅的忘却の抑制** — QLoRA (4bit) + 低ランク (r=16) で元の英語/コード/数学能力を保持
2. **Colab 1 台で完結** — T4 16GB で学習〜推論まで完遂可能
3. **データ品質の担保** — oasst1-21k-ja から構造化フィルタで 8,000 件を厳選
4. **再現性** — シード固定・ハイパーパラメータ公開・データ抽出ロジック明文化

### なぜ Qwen3.5-0.8B なのか?

| 特徴 | 詳細 |
|------|------|
| サイズ | 0.8B パラメータ (軽量・省メモリ) |
| アーキテクチャ | Gated DeltaNet + Gated Attention のハイブリッド (Qwen3.5 独自) |
| コンテキスト長 | 262,144 (ネイティブ) |
| 言語サポート | 201 言語 (日本語含む) |
| マルチモーダル | Image-Text-to-Text 対応 (本ノートブックでは vision 層を凍結し text のみ学習) |
| VRAM (LoRA bf16) | 3GB (Unsloth 計測) |

### なぜ Unsloth なのか?

- Qwen3.5 ファミリー公式サポート (0.8B / 2B / 4B / 9B / 27B / 35B-A3B / 122B-A10B)
- FA2 比で **1.5× 高速・50% VRAM 節約** (Unsloth 公式)
- `FastVisionModel` でマルチモーダルモデルの vision / language 個別学習が可能
- GGUF エクスポートまでワンストップ

---

## クイックスタート

### 必要環境

| 項目 | 推奨 |
|------|------|
| 実行環境 | Google Colab (T4 / L4 / A100 / V100) |
| GPU メモリ | 16GB 以上 (T4 で動作確認想定) |
| Python | 3.10 以上 |
| ランタイム | GPU ランタイム (T4 以上推奨) |

### 手順

1. **Colab を開く**: https://colab.research.google.com/
2. **ノートブックをアップロード**: 「ファイル → ノートブックをアップロード」で `qwen3.5-0.8b-ja-qlora.ipynb` を選択
3. **GPU ランタイムに変更**: 「ランタイム → ランタイムのタイプを変更 → T4 GPU」
4. **上から順に実行**: セルを順番に実行 (全実行で 30 〜 60 分程度)

### ローカル実行の場合

Colab 以外で実行する場合は、[Unsloth 公式のインストールガイド](https://unsloth.ai/docs/basics/install-to-own-environment) に従って Unsloth を導入してください。ノートブックのインストールセルは Colab 用のため、ローカルでは別途環境構築が必要です。

```bash
# 例: venv での準備
python -m venv venv
source venv/bin/activate
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install "trl<0.9.0" peft accelerate bitsandbytes xformers triton
```

---

## アーキテクチャと設計思想

### 破滅的忘却 (Catastrophic Forgetting) への対応

軽量モデルの日本語微調整で最も警戒すべきリスクが「元の英語/コード能力の喪失」です。
本ノートブックでは以下の複合的アプローチでこれを抑制します:

```
┌─────────────────────────────────────────────────────┐
│  破滅的忘却の抑制: 多層防御                          │
├─────────────────────────────────────────────────────┤
│  1. QLoRA (4bit)                                    │
│     └ 元重みを完全凍結、低ランク差分のみ学習         │
│                                                       │
│  2. Vision 層の凍結                                  │
│     └ finetune_vision_layers=False                  │
│                                                       │
│  3. 低ランク (r=16, alpha=32)                        │
│     └ 表現力を制限し過学習を防止                     │
│                                                       │
│  4. 短い Epochs (3)                                  │
│     └ 学習ステップを抑え、元知識への干渉を最小化     │
│                                                       │
│  5. 高品質データ (8k 厳選)                           │
│     └ 多様性 + 品質で特定パターンへの偏りを回避     │
│                                                       │
│  6. cosine + warmup                                  │
│     └ 学習率スケジュールで後半は緩やかに             │
└─────────────────────────────────────────────────────┘
```

### 学習パイプライン全体図

```
llm-jp/oasst1-21k-ja (21,200件)
       │
       ▼
┌──────────────────────┐
│  品質フィルタ          │
│  - human→gpt 交互確認 │
│  - 長さ (20-2000/50-4000) │
│  - 日本語率 (≥30%)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  シャッフル + サンプリング │
│  seed=3407, 8,000件   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  ChatML 形式化         │
│  (Qwen 標準テンプレート)│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  QLoRA 学習            │
│  FastVisionModel       │
│  + 7層 LoRA (r=16)     │
│  + paged_adamw_8bit    │
└──────────┬───────────┘
           │
           ▼
   ┌───────┴───────┐
   ▼               ▼
LoRA Adapter   Merged 16bit
(軽量・差分)    (推論用・HF Hub)
```

---

## ハイパーパラメータ

### 学習設定 (仕様書準拠)

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| 手法 | QLoRA (4bit) | 元重みを凍結し忘却を抑制 |
| LoRA Rank (`r`) | 16 | 中程度の表現力で過学習防止 |
| LoRA Alpha (`α`) | 32 | `α = 2r` でスケーリング |
| LoRA Dropout | 0 | Unsloth 公式推奨 |
| ターゲット層 | `q/k/v/o/gate/up/down_proj` | attn + MLP の全 7 層を広くカバー |
| Vision 層 | 凍結 (`finetune_vision_layers=False`) | テキストのみ学習し VRAM 節約 |
| Learning Rate | 2e-4 | LoRA 向きの中程度の学習率 |
| LR Scheduler | cosine + warmup(50 steps) | 後半は緩やかに収束 |
| Epochs | 3 | 短めに設定して過学習抑制 |
| Batch Size | 2 | T4 16GB + 4bit で 2048 は余裕 |
| Gradient Accumulation | 4 | 実効 batch = 8 |
| Max Sequence Length | 2048 | oasst1 の長さ分布に合わせる |
| Optimizer | `paged_adamw_8bit` | VRAM 節約 + 安定収束 |
| Weight Decay | 0.01 | 標準的な正則化 |
| Max Grad Norm | 1.0 | 勾配爆発防止 |
| Gradient Checkpointing | `unsloth` | 長文脈学習に必須 |
| Seed | 3407 | 再現性 |

### 推論設定

| パラメータ | 値 |
|-----------|-----|
| `do_sample` | True |
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `max_new_tokens` | 256 |

---

## データセット

### `llm-jp/oasst1-21k-ja` (実確認済)

OpenAssistant oasst1 の日本語サブセット (DeepL による英→日翻訳)。

- **HF URL**: https://huggingface.co/datasets/llm-jp/oasst1-21k-ja
- **行数**: 21,200 件
- **ライセンス**: Apache 2.0
- **形式**: ShareGPT 形式

```json
{
  "conversations": [
    {"from": "human", "value": "こんにちは！"},
    {"from": "gpt",   "value": "こんにちは！ご質問や..."},
    {"from": "human", "value": "..."},
    {"from": "gpt",   "value": "..."}
  ]
}
```

> **注意**: 列は `conversations` のみ。`rank` / `message_id` / `parent_id` は**存在しません** (元の oasst1 にはありますが、llm-jp 版では省略されています)。

### 抽出フィルタの詳細

| フィルタ | 条件 | 目的 |
|---------|------|------|
| ペア完全性 | human → gpt → human → gpt の交互構造 | 不完全対話の除外 |
| ユーザー発話長 | 20 〜 2000 文字 | 短すぎる/長すぎる除外 |
| アシスタント応答長 | 50 〜 4000 文字 | ノイズ・長文除外 |
| 日本語率 (user) | ≥ 20% | 機械翻訳ノイズ除外 |
| 日本語率 (assistant) | ≥ 30% | より厳しく |
| サンプリング | シード固定 (3407) でシャッフル後 8,000 件 | 多様性担保 |

日本語率は `[ぁ-んァ-ヶー一-龥]` の正規表現で計算します。

---

## ノートブックの構成

`qwen3.5-0.8b-ja-qlora.ipynb` は以下の 9 セクション (28 セル) で構成されています:

| # | セクション | 内容 | 想定時間 |
|---|-----------|------|---------|
| 1 | 環境セットアップ | GPU 確認 / Unsloth インストール / ハイパラ定義 | 2 分 |
| 2 | モデル読み込み | `FastVisionModel.from_pretrained` (4bit) | 3 分 |
| 3 | LoRA 適用 | `get_peft_model` (vision 凍結, 7 層ターゲット) | 1 分 |
| 4 | データセット準備 | oasst1-21k-ja ロード / フィルタ / サンプリング / ChatML 化 | 5 分 |
| 5 | 学習 | `SFTTrainer.train()` (3 epochs) | 30〜45 分 |
| 6 | 保存 | LoRA adapter + 16bit merged モデル | 2 分 |
| 7 | 推論テスト | 日本語 + 英語で忘却チェック | 1 分 |
| 8 | (任意) HF Hub アップロード | コメントアウト済 | — |
| 9 | 今後の改善方向 | ROADMAP.md への参照 | — |

---

## Qwen3.5 特有の注意点

### 1. `FastVisionModel` を使う理由

Qwen3.5-0.8B は HuggingFace 上で `Image-Text-to-Text` (マルチモーダル) として公開されています。
Unsloth では `FastLanguageModel` ではなく **`FastVisionModel`** を使う必要があります。

### 2. テキストのみ学習する方法

`FastVisionModel.get_peft_model` で以下のフラグを制御します:

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,  # ★ vision 層を凍結
    finetune_language_layers   = True,   # 言語層は学習
    finetune_attention_modules = True,   # attn (q/k/v/o) を学習
    finetune_mlp_modules       = True,   # MLP (gate/up/down) を学習
    ...
)
```

### 3. 推論能力 (`<think>` ブロック) の保持

Qwen3.5 は `<think>...</think>` 形式の推論プロセスを出力するモデルです。
oasst1 には思考プロセスが含まれないため、学習後に推論能力が変化する可能性があります。

**推奨対応** (Unsloth 公式):
- 推論スタイルのデータ (例: Open-R1, Magpie-Reasoning 等) を **25% 以上** ブレンドする
- または、直接回答スタイルのみで運用する (推論能力を諦める)

### 4. QLoRA と bf16 LoRA の使い分け

| 項目 | QLoRA (本ノートブック) | bf16 LoRA (公式推奨) |
|------|----------------------|---------------------|
| VRAM (0.8B) | 〜6GB | 3GB |
| 速度 | やや遅い | 高速 |
| 精度 | わずかに劣る場合あり | 最高 |
| 用途 | T4/16GB 等の制約環境 | VRAM に余裕がある場合 |

> Qwen3.5-27B+ の MoE モデルでは QLoRA は非推奨 (Unsloth 公式)。
> 0.8B は dense モデルなので QLoRA でも問題なく動作します。

---

## トラブルシューティング

### OOM (Out of Memory) が発生する

```
CUDA out of memory. Tried to allocate 2.00 GiB...
```

**対策** (優先順):

1. `BATCH_SIZE = 1` に下げる
2. `MAX_SEQ_LEN = 1024` に下げる
3. `GRAD_ACCUM = 8` に上げて実効 batch を維持
4. Colab Pro にアップグレード (A100 16GB 利用可)

### `FastVisionModel` が ImportError になる

```
ImportError: cannot import name 'FastVisionModel' from 'unsloth'
```

**原因**: Unsloth が古い、またはインストールに失敗している。

**対策**:

```python
# Colab で再インストール
!pip uninstall -y unsloth
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
```

### データセット読み込みでエラー

```
DatasetNotFoundError: Dataset 'llm-jp/oasst1-21k-ja' doesn't exist
```

**原因**: HF Hub の一時的な問題、またはネットワーク制限。

**対策**:

- 数分待ってからリトライ
- `load_dataset("llm-jp/oasst1-21k-ja", split="train")` を再実行

### 学習が遅い

- T4 は bf16 非対応のため fp16 で動作 → やや遅い
- `packing=True` を試すと短い対話の効率が上がる場合あり
- どうしても遅い場合は Colab Pro の A100/L4 を検討

### 学習後のモデルが英語を忘れている

破滅的忘却が起きています。以下を試してください:

1. `EPOCHS = 2` に下げる
2. 英語汎用データ (`databricks/dolly-15k` から 1k 件など) を 10% ブレンド
3. `LR = 1e-4` に下げる
4. ROADMAP.md の Phase 3「忘却抑制の強化」を参照

### チャットテンプレートが不正

```python
# 確認
print(tokenizer.chat_template)
# None の場合は ChatML を設定
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="chatml", ...)
```

---

## 評価と次のステップ

### 推奨される定量評価

学習後のモデルを定量評価することで、改善判定が明確になります。
ROADMAP.md の Phase 2 を参照してください。

| 評価ツール | 用途 | URL |
|-----------|------|-----|
| `llm-jp/llm-jp-eval` | 日本語タスク総合評価 | https://github.com/llm-jp/llm-jp-eval |
| `lm-evaluation-harness` | MMLU / MMLU-JA 等 | https://github.com/EleutherAI/lm-evaluation-harness |
| `japanese-eval` | 日本語特化ベンチ | https://github.com/ |

### 改善ロードマップ

ROADMAP.md に Phase 1 〜 5 の改善計画をまとめています:

1. **Phase 1** (DONE): ベースライン構築
2. **Phase 2** (NEXT): 定量評価の導入
3. **Phase 3**: 忘却抑制の強化 (データ混合・KL 正則化)
4. **Phase 4**: インストラクション追従の強化 (DPO/ORPO)
5. **Phase 5**: 実用化とデプロイ (GGUF / vLLM / HF Spaces)

---

## ファイル構成

```
.
├── qwen3.5-0.8b-ja-qlora.ipynb   # Colab ノートブック本体 (28 セル)
├── ROADMAP.md                     # 改善ロードマップ (Phase 1〜5)
├── README.md                      # このファイル
└── .gitignore                     # 学習成果物 (*.safetensors 等) を除外
```

### 学習後に生成されるファイル (git 管理外)

```
outputs/                # TrainingArguments の出力 (チェックポイント)
lora_adapter/           # LoRA アダプタのみ (軽量・数 MB)
merged_model_fp16/      # 16bit マージ済みモデル (推論用・1.6GB 程度)
```

---

## ライセンス

### コード

MIT License - 詳細は `LICENSE` ファイルを参照 (ノートブック・スクリプト)

### 学習済みモデル

学習済みアダプタ・マージ済みモデルを再配布する場合は、以下のライセンスを確認してください:

| コンポーネント | ライセンス | URL |
|---------------|----------|-----|
| Qwen3.5-0.8B (原モデル) | Apache 2.0 | https://huggingface.co/Qwen/Qwen3.5-0.8B |
| llm-jp/oasst1-21k-ja (データ) | Apache 2.0 | https://huggingface.co/datasets/llm-jp/oasst1-21k-ja |
| OpenAssistant oasst1 (原データ) | Apache 2.0 | https://huggingface.co/datasets/OpenAssistant/oasst1 |

### 引用

本ノートブックを参考にした場合は、以下を引用してください:

```bibtex
@misc{qwen35-ja-qlora,
  title  = {Qwen3.5-0.8B Japanese QLoRA Fine-tuning with Catastrophic-Forgetting Mitigation},
  author = {CloudP0nd},
  url    = {https://github.com/CloudP0nd/qwen3.5-0.8b-ja-qlora},
  year   = {2026}
}
```

---

## 参考資料

### 公式ドキュメント

- **Unsloth Qwen3.5 Fine-tune Guide**: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- **Unsloth Vision Fine-tuning**: https://unsloth.ai/docs/basics/vision-fine-tuning
- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **Qwen3.5 Model Card**: https://huggingface.co/Qwen/Qwen3.5-0.8B

### 論文

- **QLoRA**: https://arxiv.org/abs/2305.14314
- **LoRA**: https://arxiv.org/abs/2106.09685
- **rsLoRA**: https://arxiv.org/abs/2312.03732
- **DoRA**: https://arxiv.org/abs/2402.09353

### データセット

- **oasst1 (原)**: https://huggingface.co/datasets/OpenAssistant/oasst1
- **llm-jp/oasst1-21k-ja**: https://huggingface.co/datasets/llm-jp/oasst1-21k-ja
- **llm-jp 公式**: https://llm-jp.nii.ac.jp/

### 評価ツール

- **llm-jp-eval**: https://github.com/llm-jp/llm-jp-eval
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness

### 関連ノートブック (Unsloth 公式)

- [Qwen3.5-0.8B Vision Fine-tune](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(0_8B)_Vision.ipynb)
- [Qwen3.5-4B GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_GRPO.ipynb)

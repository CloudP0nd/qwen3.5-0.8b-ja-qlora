# Qwen3.5-0.8B 日本語 QLoRA 微調整

Qwen3.5-0.8B (FP16) を Unsloth + QLoRA で日本語強化学習する Google Colab ノートブックです。
`llm-jp/oasst1-21k-ja` から抽出した高品質 8,000 件を用い、**破滅的忘却を最小限に抑えながら** 日本語性能を底上げすることを目的としています。

## クイックスタート

1. [Colab でノートブックを開く](https://colab.research.google.com/) → 「ファイル → ノートブックをアップロード」で `qwen3.5-0.8b-ja-qlora.ipynb` を選択
2. ランタイムタイプを **GPU (T4 以上推奨)** に設定
3. 上から順にセルを実行

## 設計ハイライト

| 項目 | 値 | 根拠 |
|------|-----|------|
| 手法 | QLoRA (4bit) | 元重みを凍結し忘却を抑制 |
| LoRA Rank / Alpha | 16 / 32 | 中程度の表現力で過学習防止 |
| ターゲット層 | q/k/v/o/gate/up/down_proj | attn + MLP を広くカバー |
| Vision 層 | 凍結 (`finetune_vision_layers=False`) | テキストのみ学習し VRAM 節約 |
| LR | 2e-4 (cosine) | LoRA 向きの中程度の学習率 |
| Epochs | 3 | 短めに設定して過学習抑制 |
| Max Seq | 2048 | oasst1 の長さ分布に合わせる |
| Optimizer | paged_adamw_8bit | VRAM 節約 + 安定収束 |
| データ | oasst1-21k-ja → 8k 高品質抽出 | 多様性 + 品質両立 |

## Qwen3.5 特有の注意点

- Qwen3.5-0.8B はマルチモーダル (Vision) モデルとして公開されていますが、Unsloth 公式サポート済みです。
- テキストのみ学習するため `FastVisionModel` を使用し、`finetune_vision_layers=False` で視覚エンコーダを凍結します。
- Qwen3.5 は `<think>...</think>` 推論ブロックを持つため、推論能力を保ちたい場合はデータに推論スタイルを 25% 以上混ぜることを推奨します (Unsloth 公式)。
- 公式推奨は bf16 LoRA (3GB VRAM) ですが、0.8B は dense モデルなので QLoRA (4bit) も動作します。

## データセットスキーマ (実確認済)

`llm-jp/oasst1-21k-ja` は ShareGPT 形式です:

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt",   "value": "..."}
  ]
}
```

- 列は `conversations` のみ (rank / message_id / parent_id は存在しない)
- 行数は 21,200 件

## ファイル構成

```
.
├── qwen3.5-0.8b-ja-qlora.ipynb   # Colab ノートブック本体
├── ROADMAP.md                     # 改善ロードマップ (Phase 1〜5)
├── README.md                      # このファイル
└── .gitignore
```

## 注意事項

- Colab 無料枠の T4 (16GB) を想定しています。OOM が発生する場合は `BATCH_SIZE=1` または `MAX_SEQ_LEN=1024` に調整してください。
- 学習後のアダプタを再配布する場合は、データ/モデルのライセンスを各自確認してください。

## ライセンス

ノートブック・スクリプト: MIT
学習済みモデル: 原モデル (Qwen3.5) およびデータセット (oasst1/llm-jp) のライセンスに従ってください。

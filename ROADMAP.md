# ROADMAP — Qwen3.5-0.8B 日本語強化学習プロジェクト

本ドキュメントは、Qwen3.5-0.8B (FP16) の日本語性能を **破滅的忘却 (catastrophic forgetting)** を最小限に抑えながら底上げするためのロードマップです。
v1 は QLoRA + oasst1-21k-ja (8k 抽出) によるベースライン構築です。以降は評価→改善→拡張の順に進めます。

---

## 設計原則 (North Star)

1. **原本知識を守る**: 元モデルの英語/コード/数学能力を損なわないこと。
2. **日本語特有の表現力を高める**: 自然な文体・敬語・漢字かなバランス・指示追従性の改善。
3. **Colab 1 台で完結**: T4 16GB で学習〜推論まで完遂できるスケールを保つ。
4. **再現性**: シード固定・データ抽出ロジック明文化・ハイパーパラメータ公開。
5. **定量評価**: 主観だけでなくベンチマークで前後比較を行う。

---

## Phase 1 — ベースライン構築 (DONE)

- [x] Unsloth + QLoRA による学習パイプライン構築
- [x] `llm-jp/oasst1-21k-ja` から高品質 8,000 件を抽出するフィルタ実装
  - [x] ShareGPT 形式 (`conversations`: `[{from, value}]`) への対応 (実スキーマ確認済)
  - [x] human → gpt の交互ペア完全性チェック
  - [x] 長さフィルタ (user 20〜2000 / assistant 50〜4000 文字)
  - [x] 日本語率フィルタ (ひらがな/カタカナ/漢字 30% 以上)
  - [x] シード固定シャッフルで 8k サンプリング
- [x] LoRA 設定: r=16 / alpha=32 / 全 7 モジュール (attn + MLP)
- [x] 学習設定: LR=2e-4 / epochs=3 / paged_adamw_8bit / cosine / warmup=50
- [x] チャットテンプレート (ChatML) 適用
- [x] `FastVisionModel` + `finetune_vision_layers=False` でテキストのみ学習
- [x] 日本語 + 英語での推論サンプル (忘却チェック)

**成果物**: `qwen3.5-0.8b-ja-qlora.ipynb`

---

## Phase 2 — 定量評価の導入 (NEXT)

ベースラインが「主観的に良さそう」でも、定量的比較が無いと改善判定ができません。

- [ ] `llm-jp/llm-jp-eval` を Colab で実行可能な形に持ち込む
  - タスク: JCommonsenseQA / JNLI / MARC-ja / JSQuAD / etc.
  - スコアを `eval/baseline.json`, `eval/qlora_v1.json` として保存
- [ ] `lm-evaluation-harness` で MMLU / MMLU-JA を実行 (英語忘却の定量化)
- [ ] 同一プロンプト 100 件で生成品質を LLM-as-a-Judge 評価
  - Judge: GPT-4 / Claude 等の強モデル
  - 軸: 自然さ / 指示追従 / 事実正確性 / 日本語らしさ
- [ ] 評価結果をグラフ化し `eval/report.md` にまとめる

**完了条件**: ベースラインモデルと v1 を 5 タスク以上で数値比較できること。

---

## Phase 3 — 忘却抑制の強化

Phase 2 の評価で英語/コード性能が落ちている場合、以下を段階的に導入。

### 3.1 データ混合 (Replay)
- [ ] 英語汎用データを 10〜20% ブレンド
  - 候補: `databricks/dolly-15k`, `tatsu-lab/alpaca`, `Open-Orca/OpenOrca` (1k〜2k 件)
- [ ] コードサンプルを 5% ブレンド (`CodeAlpaca-ja` 等のサブセット)
- [ ] 混合比ごとに評価し、最適比を決定

### 3.2 学習戦略の調整
- [ ] LoRA rank / alpha のスイープ (r=8/16/32, alpha=r/2r/4r)
- [ ] LR スケジューラ比較 (cosine / linear / constant_with_warmup)
- [ ] KL 正則化 (元モデル出力との KL 距離を損失に加算) の実験
- [ ] Early Stopping (val loss 監視) の導入

### 3.3 パラメータ効率の最適化
- [ ] rsLoRA (rank-stable LoRA) の比較
- [ ] DoRA (Weight-Decomposed LoRA) の比較
- [ ] PiSSA などの初期化手法の比較

**完了条件**: MMLU/MMLU-JA のスコア低下を 2pt 未満に抑えつつ、日本語ベンチマークを +3pt 以上改善すること。

---

## Phase 4 — インストラクション追従の強化

### 4.1 データセット拡張
- [ ] `kunishou/databricks-dolly-15k-ja` を追加 (分類/クローズドブック/QA/要約)
- [ ] `llm-jp/AnswerCarefully` を追加 (日本のコンテキストに特化)
- [ ] `shisa-ai/shisa-preference-ja` で好みデータを準備

### 4.2 マルチターン対応
- [ ] oasst1 は単発中心なので、マルチターンデータ (`sharegpt-ja` 等) を追加
- [ ] 対話履歴を含めた学習フォーマットの検討

### 4.3 好みチューニング
- [ ] Phase 3 の SFT モデルをベースに DPO を実施
- [ ] ORPO (Odds Ratio Preference Optimization) の比較検討
- [ ] Iterative DPO で段階的に改善

**完了条件**: 人手評価 (ペア比較) で v1 を 60% 以上上回ること。

---

## Phase 5 — 実用化とデプロイ

- [ ] GGUF 量子化 (4bit/5bit/8bit) でのエクスポート
- [ ] llama.cpp / Ollama でのローカル推論テスト
- [ ] vLLM でのサービング検証
- [ ] Hugging Face Spaces にデモを公開
- [ ] モデルカード (Model Card) の整備
  - 学習データ詳細
  - 評価スコア一覧
  - 既知の制限事項 (limitations)
  - 使い方サンプル

**完了条件**: エンドユーザがワンコマンドで試せる状態 (Ollama pull / HF Space) になること。

---

## リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| Colab T4 OOM | 学習不能 | bs=1, max_seq=1024, または gradient_checkpointing 強化 |
| oasst1 トピック偏り | 特定分野への過学習 | シャッフル + 他データ混合 (Phase 3) |
| Qwen3.5 推論能力低下 | `<think>` ブロック消失 | 推論スタイルデータを 25% 以上ブレンド (Unsloth 公式推奨) |
| QLoRA × MoE 非推奨 | 学習不安定 | 0.8B は dense なので問題なし。27B+ の MoE を使う場合は bf16 LoRA 推奨 |
| 忘却の定量検出漏れ | 改善判定ミス | Phase 2 の評価を必ず先行 |
| ライセンス | 再配布不可 | データ/モデルのライセンスを都度確認 |

---

## マイルストーン (目安)

| フェーズ | 想定所要時間 | 備考 |
|----------|--------------|------|
| Phase 1 | 1 日 | Colab 1 セッションで学習完了 |
| Phase 2 | 2〜3 日 | 評価スクリプト整備が主作業 |
| Phase 3 | 1 週間 | ハイパーパラメータスイープ |
| Phase 4 | 1〜2 週間 | DPO 用データ準備が律速 |
| Phase 5 | 3〜5 日 | デプロイ周りが主作業 |

---

## 参考資料

- Unsloth: https://github.com/unslothai/unsloth
- QLoRA 論文: https://arxiv.org/abs/2305.14314
- oasst1: https://huggingface.co/datasets/OpenAssistant/oasst1
- llm-jp/oasst1-21k-ja: https://huggingface.co/datasets/llm-jp/oasst1-21k-ja
- llm-jp-eval: https://github.com/llm-jp/llm-jp-eval
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- DoRA: https://arxiv.org/abs/2402.09353
- rsLoRA: https://arxiv.org/abs/2312.03732

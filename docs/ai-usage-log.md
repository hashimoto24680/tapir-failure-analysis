# 生成AI利用ログ

## 2026-07-21〜2026-07-22

- 使用AI: OpenAI Codex
- 主な用途: 課題条件の整理、候補論文の一次情報調査、PCスペックと再現可能性の比較、TAPIRの選定、初期実験計画とプロジェクト構成の作成
- 自分で判断した箇所: 対象論文をTAPIRに決定
- 今後記録する内容: 環境構築、実装支援、failure条件設計、評価コード、改善案、レポート推敲におけるAI利用と、人間が修正・判断した箇所

## 2026-07-22〜2026-07-23 環境構築とbaseline smoke test

- 使用AI: OpenAI Codex
- 主な用途: 既存状態の監査、Blackwell対応PyTorchの公式配布確認、Python 3.11仮想環境の構築、公式TapNet source/checkpointの特定と固定、CUDA/TAPIR smoke test・合成入力・可視化・再現スクリプト・テストの実装、コードレビュー、再現性記録と文書の整合確認
- AIが発見して回避した点: 現行の公式`torch_tapir_demo.ipynb`は名称と異なりBootsTAPIR checkpointを使う。ICCV 2023 paper baselineとして`tapir_checkpoint_panning.pt`、`pyramid_level=0`、`extra_convs=False`を選び、checkpoint SHA-256とstrict state-dict loadで検証した
- 実行結果: PyTorch 2.13.0+cu130で`sm_120` CUDA検査成功。16-frame・5-point合成clipでEPE 0.887 px、PCK@3/5px 1.0、低GPU負荷時のwarm-up後0.065秒、PyTorch peak allocated約483 MB
- 人間が今後判断する箇所: Git commitに使う氏名・メール、failure条件の最終強度範囲、tuning/final split、レポートに採用する比較表と考察

## 2026-07-23 GitHub公開準備

- 使用AI: OpenAI Codex
- ユーザー方針: 本人のGitHubへ最初はprivate repositoryとして保存し、課題提出時にpublicへ変更する
- 主な用途: commit対象の列挙、秘密情報・大容量ファイル・外部データの混入確認、公開時に不要なローカル絶対パスの除去、GitHub CLI・接続アカウントの確認
- 除外を確認した内容: checkpoint、公式TapNet checkout、生成動画・NPZ、仮想環境、外部データ

## 2026-07-23 TAPIR論文の初心者向け理解支援

- 使用AI: OpenAI Codex
- 主な用途: ICCV 2023論文と公式プロジェクトの一次情報確認、Tracking Any Point・二段階構成・遮蔽判定・評価指標の初心者向け説明への言い換え、現在のfailure分析計画との対応付け
- AIが整理した要点: TAPIRは各フレームでの大域的な候補探索と、時間方向の情報を使う局所的な反復修正を組み合わせ、遮蔽後の再発見と滑らかな高精度追跡の両立を狙う
- 人間が今後判断する箇所: 実際に理解しにくかった用語、レポートで重点的に説明する機構、failure条件と改善案の最終選択

## 2026-07-26 TAPIRの特徴量とtemporal refinementの実装確認

- 使用AI: OpenAI Codex
- 主な用途: 固定済みの公式Standard PyTorch実装を読み、ResNet特徴マップ、query特徴の補間、cost volume、局所相関、時間方向convolution、反復更新の関係を初心者向けに説明
- AIが確認した要点: 256×256入力ではstride 4の128次元特徴とstride 8の256次元特徴を使う。初期探索はquery特徴と全候補点の内積、refinementは現在位置周辺の7×7局所相関を両解像度で求め、時間方向ネットワークから位置・遮蔽・不確実性・内部特徴の更新量を出す
- 人間が今後判断する箇所: 特徴量・局所相関・時間的整合性のうち、レポート本文でどこまで実装詳細を説明するか

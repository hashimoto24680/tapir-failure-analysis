# TAPIR Failure Case Analysis

ICCV 2023 main-track full paper **“TAPIR: Tracking Any Point with Per-Frame Initialization and Temporal Refinement”** を再現し、failure case の分析と改善を行うプロジェクトです。

- Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.html
- Official repository: https://github.com/google-deepmind/tapnet
- Selected model: **Standard PyTorch TAPIR corresponding to the ICCV 2023 paper**

現行の公式リポジトリには BootsTAPIR、TAPNext などの後続モデルも含まれます。ベースラインでは、それらをTAPIRとして扱いません。使用するcommit、設定、checkpointを固定して記録します。

## Project goal

1. 公開済みTAPIR重みでローカル推論を再現する。
2. 少なくとも2種類のfailure conditionを、強度を制御して実験する。
3. failure detectorまたは再初期化処理を1つ以上実装する。
4. 同一入力上で改善前後を定量・定性比較する。
5. 4〜8ページのレポート、ソースコード、生成AI利用ログを提出可能な形にする。

## Planned experiment

- Failure A: 無地・低コントラスト・周期模様
- Failure B: 長時間遮蔽・motion blur・再出現
- Optional Failure C: シーンカットまたは画面外退出
- Improvement: forward/backward cycle consistency と予測信頼度によるfailure detection、必要時のみ再初期化
- Metrics: EPE、PCK@3/5px、遮蔽判定F1、遮蔽後復帰率、accuracy–coverage、処理時間、最大VRAM

詳細は [docs/experiment-plan.md](docs/experiment-plan.md) を参照してください。

## Baseline guardrail

現行の公式 `torch_tapir_demo.ipynb` は、表示名がStandard PyTorch TAPIRでも、実際にはBootsTAPIR checkpointを読み込みます。本プロジェクトではICCV 2023 baselineを次の組み合わせへ固定しています。

- TapNet commit: `989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc`
- checkpoint: `tapir_checkpoint_panning.pt`
- checkpoint SHA-256: `628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa`
- model: `TAPIR(pyramid_level=0, extra_convs=False)`
- inference: whole-video、256×256、float32

`TAPIR()` の現行デフォルトはBootsTAPIR側の構成なので使用しません。ローダーはcheckpoint hashと`strict=True`のstate-dict loadを検証します。

## Setup

Windows PowerShellで次を実行します。Python 3.11仮想環境、CUDA 13.0版PyTorch、固定commitの公式TapNet、正しいcheckpoint、CUDA検査までを再現します。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

セットアップ後の検証:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run_tapir_smoke.py
```

記録は `results/setup/`、動画・接触シート・生の追跡出力は `results/generated/smoke/` に生成されます。後者とcheckpoint、公式source checkoutはGit対象外であり、`scripts/setup.ps1` から再取得できます。

現在の環境構築結果と測定値は [docs/setup-record.md](docs/setup-record.md)、実験設計は [docs/experiment-plan.md](docs/experiment-plan.md) を参照してください。

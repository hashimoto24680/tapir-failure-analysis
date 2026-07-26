# Experiment plan

## Research question

TAPIRは、どのような視覚条件・遮蔽条件で点追跡に失敗し、その失敗をforward/backward consistencyと予測信頼度から事前検出できるか。

## Baseline

- Official Standard PyTorch TAPIR corresponding to ICCV 2023
- Public pretrained checkpoint
- Whole-video inference on short clips
- Initial target: 256×256または512×512、30〜120 frames、batch size 1

固定した再現条件:

- official TapNet commit: `989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc`
- checkpoint: `tapir_checkpoint_panning.pt`
- checkpoint SHA-256: `628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa`
- architecture: `pyramid_level=0`、`extra_convs=False`、`initial_resolution=(256, 256)`
- 入力動画: RGB、`[T,H,W,3]`、uint8をfloat32の`[-1,1]`へ変換
- query座標: `(time, y, x)`、出力track座標: `(x, y)`
- visibility: 公式PyTorch notebookと同じocclusion/expected-distance閾値

256×256をpaper baselineとする。512×512を使う場合は解像度変更実験として分けて報告する。

## Data

### Controlled synthetic clips

既知の点座標から正解軌跡を計算できるよう、静止画像または簡単な図形に対して平行移動、回転、拡大縮小を適用する。以下を独立に段階化する。

- texture contrast
- repeated-pattern frequency
- occlusion duration
- motion-blur kernel length
- target displacement per frame

### Real clips

公式評価データの小規模subset、または本人作成・同意取得済み動画を使用する。データ本体は原則Gitへ含めず、取得手順と固定IDのみを記録する。

## Failure conditions

### A. Appearance ambiguity

- 無地面
- 低コントラスト
- checkerboardなどの周期模様

仮説: 局所的に識別可能な特徴が不足または重複し、誤った候補点へ移る。

### B. Visibility and temporal discontinuity

- 遮蔽時間を段階的に増加
- 遮蔽中または再出現直後にmotion blurを追加
- 任意で画面外退出・シーンカットも検証

仮説: 長時間観測されない間に位置・特徴の不確実性が増え、再出現時の対応付けに失敗する。

## Improvement

1. 元動画を順方向に追跡する。
2. 終端から逆方向にも追跡する。
3. 開始点への復帰誤差とモデル信頼度をfailure scoreへ変換する。
4. 高リスク点を警告・棄却する。
5. 長時間遮蔽後のみ再初期化する方式を追加比較する。

すべてを棄却すれば精度が上がる問題を避けるため、accuracy–coverage曲線を必ず報告する。

## Metrics

- Endpoint Error (EPE)
- PCK@3px / PCK@5px
- occlusion precision / recall / F1
- recovery success rate after occlusion
- failure-detection AUROCまたはAUPRC
- accuracy versus coverage
- inference time after warm-up
- peak GPU memory

## Reproducibility record

- OS、GPU、driver
- Python、PyTorch、CUDA runtime、依存package versions
- official repository commit
- checkpoint URLおよびSHA-256
- random seed
- input clip IDs、resolution、frame count
- corruption parameters
- metric implementation and coordinate convention

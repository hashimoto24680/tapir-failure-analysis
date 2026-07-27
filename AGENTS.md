# Project instructions

- 対象論文はICCV 2023のTAPIRに固定する。
- ベースラインには公式のStandard PyTorch TAPIR architecture/checkpointを用いる。BootsTAPIR、TAPNext、TAPNext++、CoTrackerを無断で置き換えない。
- Python 3.11のプロジェクト専用仮想環境を使う。
- GPUはWindows 11上のRTX 5060 Ti 16GB（Blackwell、sm_120）。古いPyTorch/CUDAの固定指定をそのまま導入せず、現行のBlackwell対応PyTorchで最小CUDAテストを先に行う。
- 大規模学習は行わない。公開重みの推論、評価、軽量な後処理またはfailure detectorを中心にする。
- repo commit、checkpoint URL/hash、Python/package versions、seed、入力解像度、フレーム数、精度形式、最大VRAM、推論時間を記録する。
- 閾値調整用データと最終評価用データを分け、棄却型改善ではaccuracyとcoverageを同時に報告する。
- データの利用条件を確認し、著作権や肖像権が不明な動画をリポジトリへ含めない。合成動画または本人作成・同意取得済み素材を優先する。
- API呼び出しだけで完結させず、モデル入出力、追跡結果、評価処理、改善処理をローカルコードとして明示する。

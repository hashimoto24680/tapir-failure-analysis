# Data policy

データ本体は原則としてGitへ含めません。次を保存します。

- dataset名と公式URL
- licenseまたは利用条件
- download/preparation手順
- 評価に使った固定clip ID
- 自作データの場合は作成条件と同意の有無

正解軌跡を持つ合成動画の生成コードは `scripts/` へ保存します。

## Current smoke-test data

`src/tapir_failure_analysis/synthetic.py` がseed 20260722から手続き的に生成する256×256の図形動画だけを使用しています。外部動画、人物画像、第三者の著作物は含みません。生成動画本体は `results/generated/` に置き、Gitには含めません。

# Setup and baseline record

## Reproducible baseline

| Item | Fixed value |
|---|---|
| Paper | ICCV 2023 TAPIR |
| Official repository | `https://github.com/google-deepmind/tapnet.git` |
| Source commit | `989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc` |
| Checkpoint URL | `https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt` |
| Checkpoint SHA-256 | `628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa` |
| Checkpoint size | 124,348,474 bytes |
| Architecture | `pyramid_level=0`, `extra_convs=False`, `softmax_temperature=20.0` |
| Inference path | Standard whole-video PyTorch TAPIR |

現行の公式`colabs/torch_tapir_demo.ipynb`は`bootstapir_checkpoint_v2.pt`と`pyramid_level=1`を使う。そのまま実行するとICCV 2023 paper checkpointではなくなるため、本プロジェクトの`baseline.py`はpaper checkpointのhashとarchitectureを明示的に検証する。

## Environment

- OS: Windows 11 Pro 64-bit、build 26200
- GPU: NVIDIA GeForce RTX 5060 Ti 16 GB、compute capability 12.0
- driver: 610.74
- Python: 3.11.9 (`.venv`)
- PyTorch: 2.13.0+cu130
- torchvision: 0.28.0+cu130
- CUDA runtime bundled with wheel: 13.0
- cuDNN: 92000
- precision: float32、TF32 disabled for the recorded TAPIR smoke run
- exact environment: `scripts/setup.ps1` pins pip, CUDA PyTorch/torchvision, the project editable install, TapNet source registration/commit and checkpoint; `requirements-lock.txt` pins all remaining installed packages

TapNet upstream declares its JAX/training/demo stack as base package dependencies. This project uses only the official `tapnet.torch.tapir_model`, `nets` and `utils` inference subset, so the setup script registers the clean pinned source checkout with a venv `.pth` file instead of installing inconsistent unused dependencies. `pip check` must pass at the end of setup.

CUDA 4096×4096 float32 matrix multiplication succeeded. The wheel reports `sm_120` in `torch.cuda.get_arch_list()`. Full values are in `results/setup/cuda-smoke.json`.

## TAPIR smoke result

- input: deterministic locally generated RGB clip、256×256、16 frames、5 query points、seed 20260722
- query coordinates: `(time, y, x)`
- output track coordinates: `(x, y)`
- strict checkpoint load: success、188 state keys、31,072,327 parameters
- warm-up: 0.514 seconds
- inference after warm-up: 0.065 seconds
- PyTorch peak allocated VRAM: 482,909,696 bytes
- PyTorch peak reserved VRAM: 526,385,152 bytes
- EPE: 0.887 px
- PCK@3px / PCK@5px: 1.0 / 1.0
- output video SHA-256: `21e3cf9b629d22c8df7da3447be2e014a4e9b7263e798a16d48911335caf896e`

Full values are in `results/setup/tapir-smoke.json`. The MP4, contact sheet and NPZ are generated under `results/generated/smoke/` and excluded from Git.

This run verifies compatibility and correct model selection. At inference start, `nvidia-smi` reported 1937 MiB device memory used and 3% utilization, substantially cleaner than the first compatibility run. The 0.065-second value is the recorded smoke-test time, but final report timing should still use repeated trials with other GPU-heavy applications closed.

## Repeat commands

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\run_tapir_smoke.py
```

`external/tapnet/` and `checkpoints/` are intentionally untracked. The setup script runs CUDA validation immediately after installing PyTorch, then verifies the exact source commit, clean checkout and checkpoint hash.

"""Run a minimal CUDA compatibility test and emit a reproducibility record."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip().splitlines()[0]


def run_smoke_test(matrix_size: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "compiled_arch_list": torch.cuda.get_arch_list(),
        "nvidia_driver": _driver_version(),
        "matrix_size": matrix_size,
        "precision": "float32",
    }

    if not torch.cuda.is_available():
        record["success"] = False
        record["error"] = "torch.cuda.is_available() returned False"
        return record

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    record.update(
        {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_vram_bytes": properties.total_memory,
        }
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    left = torch.randn((matrix_size, matrix_size), device=device)
    right = torch.randn((matrix_size, matrix_size), device=device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    product = left @ right
    checksum = product.float().mean().item()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    record.update(
        {
            "elapsed_seconds": elapsed,
            "result_mean": checksum,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "success": bool(torch.isfinite(product).all().item()),
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    record = run_smoke_test(args.matrix_size)
    rendered = json.dumps(record, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if record["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

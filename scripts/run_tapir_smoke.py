"""Run the official ICCV 2023 TAPIR checkpoint on a deterministic tiny clip."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from tapir_failure_analysis.baseline import (
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    MODEL_CONFIG,
    TAPNET_COMMIT,
    TAPNET_REPOSITORY,
    infer,
    load_model,
    verify_tapnet_source,
)
from tapir_failure_analysis.synthetic import SyntheticClip, make_translation_clip


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _gpu_snapshot() -> dict[str, str] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        driver, memory_used, utilization = (
            item.strip() for item in completed.stdout.splitlines()[0].split(",")
        )
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None
    return {
        "driver": driver,
        "device_memory_used_mib": memory_used,
        "device_utilization_percent": utilization,
    }


def _draw_frame(
    rgb: np.ndarray,
    frame_index: int,
    predicted_xy: np.ndarray,
    predicted_visible: np.ndarray,
    target_xy: np.ndarray,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for point_index, (prediction, target) in enumerate(
        zip(predicted_xy, target_xy, strict=True)
    ):
        truth = tuple(np.rint(target).astype(int))
        estimate = tuple(np.rint(prediction).astype(int))
        cv2.drawMarker(
            bgr,
            truth,
            (60, 220, 60),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
        color = (255, 80, 220) if predicted_visible[point_index] else (40, 40, 235)
        cv2.circle(bgr, estimate, 5, color, 2, lineType=cv2.LINE_AA)
        cv2.putText(
            bgr,
            str(point_index),
            (estimate[0] + 6, estimate[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        bgr,
        f"frame {frame_index:02d}  green=GT  magenta=TAPIR",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return bgr


def _write_visualizations(
    clip: SyntheticClip,
    tracks_xy: np.ndarray,
    visible: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "tapir-smoke.mp4"
    height, width = clip.frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")

    rendered: list[np.ndarray] = []
    try:
        for frame_index, frame in enumerate(clip.frames):
            annotated = _draw_frame(
                frame,
                frame_index,
                tracks_xy[:, frame_index],
                visible[:, frame_index],
                clip.target_tracks_xy[:, frame_index],
            )
            writer.write(annotated)
            rendered.append(annotated)
    finally:
        writer.release()

    contact_sheet_path = output_dir / "tapir-smoke-contact-sheet.png"
    selected = [rendered[0], rendered[len(rendered) // 2], rendered[-1]]
    if not cv2.imwrite(str(contact_sheet_path), np.concatenate(selected, axis=1)):
        raise RuntimeError("OpenCV could not write the contact sheet")
    return video_path, contact_sheet_path


def _metrics(
    tracks_xy: np.ndarray, visible: np.ndarray, clip: SyntheticClip
) -> dict[str, Any]:
    errors = np.linalg.norm(tracks_xy - clip.target_tracks_xy, axis=-1)
    return {
        "epe_pixels": float(errors.mean()),
        "median_error_pixels": float(np.median(errors)),
        "max_error_pixels": float(errors.max()),
        "pck_3px": float((errors <= 3.0).mean()),
        "pck_5px": float((errors <= 5.0).mean()),
        "predicted_visible_fraction": float(visible.mean()),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "checkpoints" / "tapir_checkpoint_panning.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results" / "generated" / "smoke"
    )
    parser.add_argument(
        "--record", type=Path, default=root / "results" / "setup" / "tapir-smoke.json"
    )
    args = parser.parse_args()
    checkpoint = _resolve_path(args.checkpoint, root)
    output_dir = _resolve_path(args.output_dir, root)
    record_path = _resolve_path(args.record, root)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    clip = make_translation_clip(seed=args.seed, num_frames=args.frames)
    tapnet_root = root / "external" / "tapnet"
    source_info = verify_tapnet_source(tapnet_root)
    source_info["checkout"] = _display_path(Path(str(source_info["checkout"])), root)
    source_info["imported_module"] = _display_path(
        Path(str(source_info["imported_module"])), root
    )
    model = load_model(checkpoint, device, expected_tapnet_root=tapnet_root)
    gpu_before = _gpu_snapshot()
    result = infer(
        model,
        clip.frames,
        clip.query_points_tyx,
        device,
        warmup_runs=args.warmup_runs,
    )
    gpu_after = _gpu_snapshot()

    expected_shape = clip.target_tracks_xy.shape
    if result.tracks_xy.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected track shape {result.tracks_xy.shape}; expected {expected_shape}"
        )
    video_path, contact_sheet_path = _write_visualizations(
        clip, result.tracks_xy, result.visible, output_dir
    )
    npz_path = output_dir / "tapir-smoke-outputs.npz"
    np.savez_compressed(
        npz_path,
        query_points_tyx=clip.query_points_tyx,
        target_tracks_xy=clip.target_tracks_xy,
        tracks_xy=result.tracks_xy,
        target_visible=clip.target_visible,
        visible=result.visible,
        occlusion_logits=result.occlusion_logits,
        expected_distance_logits=result.expected_distance_logits,
    )

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "success": True,
        "baseline": "ICCV 2023 Standard PyTorch TAPIR (whole-video)",
        "source": {
            **source_info,
            "expected_repository": TAPNET_REPOSITORY,
            "expected_commit": TAPNET_COMMIT,
            "project_files_sha256": {
                "baseline.py": _sha256(
                    root / "src" / "tapir_failure_analysis" / "baseline.py"
                ),
                "synthetic.py": _sha256(
                    root / "src" / "tapir_failure_analysis" / "synthetic.py"
                ),
                "run_tapir_smoke.py": _sha256(Path(__file__).resolve()),
            },
        },
        "checkpoint": {
            "url": CHECKPOINT_URL,
            "sha256": CHECKPOINT_SHA256,
            "size_bytes": checkpoint.stat().st_size,
            "strict_state_dict_load": True,
        },
        "model": {
            "config": MODEL_CONFIG,
            "state_key_count": len(model.state_dict()),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
            "precision": "float32",
            "tf32": False,
        },
        "input": {
            "seed": args.seed,
            "frame_count": len(clip.frames),
            "resolution_hw": list(clip.frames.shape[1:3]),
            "query_count": len(clip.query_points_tyx),
            "query_coordinate_order": "(time, y, x)",
            "track_coordinate_order": "(x, y)",
        },
        "timing": {
            "warmup_runs": args.warmup_runs,
            "warmup_seconds": list(result.warmup_seconds),
            "inference_after_warmup_seconds": result.inference_seconds,
        },
        "memory": {
            "allocated_before_inference_bytes": result.allocated_before_bytes,
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
            "nvidia_smi_before": gpu_before,
            "nvidia_smi_after": gpu_after,
        },
        "metrics": _metrics(result.tracks_xy, result.visible, clip),
        "outputs": {
            "video": _display_path(video_path, root),
            "video_sha256": _sha256(video_path),
            "contact_sheet": _display_path(contact_sheet_path, root),
            "npz": _display_path(npz_path, root),
        },
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict loader and inference path for the ICCV 2023 paper TAPIR model."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tapnet.torch import tapir_model

TAPNET_REPOSITORY = "https://github.com/google-deepmind/tapnet.git"
TAPNET_COMMIT = "989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc"
CHECKPOINT_URL = (
    "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt"
)
CHECKPOINT_SHA256 = (
    "628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa"
)
MODEL_CONFIG = {
    "bilinear_interp_with_depthwise_conv": False,
    "pyramid_level": 0,
    "initial_resolution": [256, 256],
    "extra_convs": False,
    "softmax_temperature": 20.0,
    "use_casual_conv": False,
}


def _find_git() -> str:
    candidates = [
        shutil.which("git"),
        str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/cmd/git.exe"),
        str(
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs/Git/cmd/git.exe"
        ),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("Git is required to verify the pinned TapNet source")


def _run_git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            [_find_git(), "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "unknown Git error").strip()
        raise RuntimeError(f"Could not verify TapNet source: {detail}") from error
    return completed.stdout.strip()


def verify_tapnet_source(expected_root: Path | None = None) -> dict[str, object]:
    """Verify that the imported model comes from the clean pinned official checkout."""
    imported_file = Path(tapir_model.__file__).resolve()
    repository = imported_file.parents[2]
    if expected_root is not None and repository != expected_root.resolve():
        raise RuntimeError(
            f"TapNet was imported from {repository}, expected {expected_root.resolve()}"
        )

    actual_commit = _run_git(repository, "rev-parse", "HEAD")
    if actual_commit != TAPNET_COMMIT:
        raise RuntimeError(
            f"TapNet commit mismatch: expected {TAPNET_COMMIT}, got {actual_commit}"
        )
    tracked_or_untracked_changes = _run_git(repository, "status", "--porcelain")
    if tracked_or_untracked_changes:
        raise RuntimeError("TapNet checkout is not clean; refuse to label it official")
    remote = _run_git(repository, "remote", "get-url", "origin")
    if remote.rstrip("/") != TAPNET_REPOSITORY.rstrip("/"):
        raise RuntimeError(
            f"TapNet origin mismatch: expected {TAPNET_REPOSITORY}, got {remote}"
        )
    return {
        "repository": remote,
        "commit": actual_commit,
        "clean": True,
        "checkout": str(repository),
        "imported_module": str(imported_file),
    }


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """CPU outputs and device-side measurements from one inference run."""

    tracks_xy: np.ndarray
    visible: np.ndarray
    occlusion_logits: np.ndarray
    expected_distance_logits: np.ndarray
    warmup_seconds: tuple[float, ...]
    inference_seconds: float
    allocated_before_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    checkpoint: Path,
    device: torch.device,
    *,
    expected_tapnet_root: Path | None = None,
) -> torch.nn.Module:
    """Load only the paper TAPIR architecture and reject any other weights."""
    verify_tapnet_source(expected_tapnet_root)
    checkpoint = checkpoint.resolve()
    actual_hash = sha256_file(checkpoint)
    if actual_hash != CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, "
            f"got {actual_hash}"
        )

    model = tapir_model.TAPIR(
        bilinear_interp_with_depthwise_conv=False,
        pyramid_level=0,
        initial_resolution=(256, 256),
        extra_convs=False,
        softmax_temperature=20.0,
        use_casual_conv=False,
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def preprocess_frames(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 THWC RGB frames to float32 BTHWC in [-1, 1]."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [T,H,W,3], got {frames.shape}")
    if frames.dtype != np.uint8:
        raise TypeError(f"Expected uint8 frames, got {frames.dtype}")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(device=device)
    return tensor.to(dtype=torch.float32).div_(255.0).mul_(2.0).sub_(1.0)[None]


def preprocess_queries(query_points_tyx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert N-by-3 (time, y, x) queries to float32 batched tensors."""
    if query_points_tyx.ndim != 2 or query_points_tyx.shape[-1] != 3:
        raise ValueError(
            "Expected query points with shape [N,3] in (time,y,x) order, "
            f"got {query_points_tyx.shape}"
        )
    return torch.as_tensor(query_points_tyx, dtype=torch.float32, device=device)[None]


def postprocess_visibility(
    occlusion_logits: torch.Tensor, expected_distance_logits: torch.Tensor
) -> torch.Tensor:
    """Apply the threshold used by the official PyTorch TAPIR notebook."""
    return (
        (1.0 - torch.sigmoid(occlusion_logits))
        * (1.0 - torch.sigmoid(expected_distance_logits))
        > 0.5
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer(
    model: torch.nn.Module,
    frames: np.ndarray,
    query_points_tyx: np.ndarray,
    device: torch.device,
    warmup_runs: int = 1,
) -> InferenceResult:
    """Run whole-video float32 inference and record time and PyTorch VRAM."""
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    video = preprocess_frames(frames, device)
    queries = preprocess_queries(query_points_tyx, device)
    warmup_times: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _synchronize(device)
            start = time.perf_counter()
            warmup_output = model(video, queries)
            _synchronize(device)
            warmup_times.append(time.perf_counter() - start)
            del warmup_output

        if device.type == "cuda":
            allocated_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            allocated_before = 0

        _synchronize(device)
        start = time.perf_counter()
        outputs = model(video, queries)
        _synchronize(device)
        elapsed = time.perf_counter() - start

        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            peak_allocated = 0
            peak_reserved = 0

        tracks = outputs["tracks"][0]
        occlusion = outputs["occlusion"][0]
        expected_distance = outputs["expected_dist"][0]
        visible = postprocess_visibility(occlusion, expected_distance)

    tensors = (tracks, occlusion, expected_distance)
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise FloatingPointError("TAPIR returned a non-finite output")

    return InferenceResult(
        tracks_xy=tracks.detach().cpu().numpy(),
        visible=visible.detach().cpu().numpy(),
        occlusion_logits=occlusion.detach().cpu().numpy(),
        expected_distance_logits=expected_distance.detach().cpu().numpy(),
        warmup_seconds=tuple(warmup_times),
        inference_seconds=elapsed,
        allocated_before_bytes=allocated_before,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )

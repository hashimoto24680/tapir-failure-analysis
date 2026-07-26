"""Small deterministic clips with exact point trajectories for smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SyntheticClip:
    frames: np.ndarray
    query_points_tyx: np.ndarray
    target_tracks_xy: np.ndarray
    target_visible: np.ndarray


def make_translation_clip(
    *,
    seed: int = 20260722,
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
) -> SyntheticClip:
    """Translate a textured patch over a static textured background."""
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if height != 256 or width != 256:
        raise ValueError("The baseline smoke clip is fixed to 256x256")

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    background = np.stack(
        (
            45 + (xx * 35 // width),
            55 + (yy * 30 // height),
            65 + ((xx + yy) * 20 // (height + width)),
        ),
        axis=-1,
    ).astype(np.int16)
    background += rng.integers(-5, 6, size=(height, width, 1), dtype=np.int16)
    background = np.clip(background, 0, 255).astype(np.uint8)

    patch_size = 64
    py, px = np.mgrid[:patch_size, :patch_size]
    checker = ((px // 8 + py // 8) % 2)[..., None]
    patch = np.where(
        checker == 0,
        np.array([225, 80, 45], dtype=np.uint8),
        np.array([35, 175, 225], dtype=np.uint8),
    )
    patch = patch.copy()
    patch[24:40, 24:40] = np.array([250, 245, 60], dtype=np.uint8)
    patch[8:16, 44:52] = np.array([245, 245, 245], dtype=np.uint8)
    patch[[0, -1], :, :] = 255
    patch[:, [0, -1], :] = 255

    start_x, start_y = 48, 82
    velocity_x, velocity_y = 4, 1
    max_frames = 1 + (width - patch_size - start_x) // velocity_x
    if num_frames > max_frames:
        raise ValueError(
            f"num_frames must not exceed {max_frames} for this fixed translation"
        )
    frames = np.repeat(background[None], num_frames, axis=0)
    for frame_index in range(num_frames):
        left = start_x + velocity_x * frame_index
        top = start_y + velocity_y * frame_index
        frames[frame_index, top : top + patch_size, left : left + patch_size] = patch

    local_points_xy = np.array(
        [[32, 32], [12, 12], [48, 12], [20, 48]], dtype=np.float32
    )
    moving_tracks = np.empty(
        (len(local_points_xy), num_frames, 2), dtype=np.float32
    )
    for frame_index in range(num_frames):
        offset = np.array(
            [
                start_x + velocity_x * frame_index,
                start_y + velocity_y * frame_index,
            ],
            dtype=np.float32,
        )
        moving_tracks[:, frame_index] = local_points_xy + offset

    static_xy = np.array([210.0, 40.0], dtype=np.float32)
    static_track = np.repeat(static_xy[None, None], num_frames, axis=1)
    target_tracks = np.concatenate((moving_tracks, static_track), axis=0)
    target_visible = np.ones(target_tracks.shape[:2], dtype=bool)

    query_points = np.column_stack(
        (
            np.zeros(len(target_tracks), dtype=np.float32),
            target_tracks[:, 0, 1],
            target_tracks[:, 0, 0],
        )
    )
    return SyntheticClip(
        frames=frames,
        query_points_tyx=query_points,
        target_tracks_xy=target_tracks,
        target_visible=target_visible,
    )

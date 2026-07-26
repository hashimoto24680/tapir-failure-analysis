from __future__ import annotations

import numpy as np
import pytest

from tapir_failure_analysis.synthetic import make_translation_clip


def test_translation_clip_is_deterministic_and_has_exact_tracks() -> None:
    first = make_translation_clip(seed=7, num_frames=4)
    second = make_translation_clip(seed=7, num_frames=4)
    np.testing.assert_array_equal(first.frames, second.frames)
    assert first.frames.shape == (4, 256, 256, 3)
    assert first.query_points_tyx.shape == (5, 3)
    assert first.target_tracks_xy.shape == (5, 4, 2)
    np.testing.assert_array_equal(
        first.target_tracks_xy[:4, 1] - first.target_tracks_xy[:4, 0],
        np.tile(np.array([4.0, 1.0]), (4, 1)),
    )
    np.testing.assert_array_equal(
        first.target_tracks_xy[-1, 1], first.target_tracks_xy[-1, 0]
    )


def test_translation_clip_rejects_frames_that_leave_the_canvas() -> None:
    with pytest.raises(ValueError, match="must not exceed 37"):
        make_translation_clip(num_frames=38)

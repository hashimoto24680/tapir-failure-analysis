from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tapir_failure_analysis.baseline import (
    TAPNET_COMMIT,
    postprocess_visibility,
    preprocess_frames,
    preprocess_queries,
    verify_tapnet_source,
)


def test_preprocess_frames_maps_uint8_to_expected_range() -> None:
    frames = np.array([[[[0, 127, 255]]]], dtype=np.uint8)
    actual = preprocess_frames(frames, torch.device("cpu"))
    assert actual.shape == (1, 1, 1, 1, 3)
    np.testing.assert_allclose(
        actual.numpy(), [[[[[-1.0, 127 / 255 * 2 - 1, 1.0]]]]], rtol=0, atol=1e-6
    )


def test_preprocess_queries_preserves_tyx_order() -> None:
    queries = np.array([[2, 10, 20]], dtype=np.float32)
    actual = preprocess_queries(queries, torch.device("cpu"))
    assert actual.tolist() == [[[2.0, 10.0, 20.0]]]


def test_visibility_matches_official_threshold() -> None:
    occlusion = torch.tensor([[-10.0, 10.0]])
    expected_distance = torch.tensor([[-10.0, -10.0]])
    assert postprocess_visibility(occlusion, expected_distance).tolist() == [
        [True, False]
    ]


def test_imported_tapnet_is_the_clean_pinned_checkout() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = verify_tapnet_source(project_root / "external" / "tapnet")
    assert source["commit"] == TAPNET_COMMIT
    assert source["clean"] is True

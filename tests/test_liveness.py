# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
import pytest

from uniface.liveness import (
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    BlinkDetector,
    compute_eye_aspect_ratios,
    eye_aspect_ratio,
)


def make_eye(opening: float, width: float = 10.0) -> np.ndarray:
    """Build 6 eye points (corner, top, top, corner, bottom, bottom) with a given opening."""
    half = opening / 2.0
    return np.array(
        [
            [0.0, 0.0],
            [width / 3.0, -half],
            [2.0 * width / 3.0, -half],
            [width, 0.0],
            [2.0 * width / 3.0, half],
            [width / 3.0, half],
        ],
        dtype=np.float32,
    )


def make_mesh(opening: float) -> np.ndarray:
    """Build a (478, 2) mesh whose two eyes have the given vertical opening."""
    mesh = np.zeros((478, 2), dtype=np.float32)
    mesh[list(LEFT_EYE_EAR_INDICES)] = make_eye(opening) + np.array([60.0, 0.0], dtype=np.float32)
    mesh[list(RIGHT_EYE_EAR_INDICES)] = make_eye(opening)
    return mesh


OPEN_EYE = 6.0  # EAR = 0.6 with width 10
CLOSED_EYE = 1.0  # EAR = 0.1


class TestEyeAspectRatio:
    def test_open_eye_has_high_ear(self):
        assert eye_aspect_ratio(make_eye(OPEN_EYE)) == pytest.approx(0.6, abs=1e-3)

    def test_closed_eye_has_low_ear(self):
        assert eye_aspect_ratio(make_eye(CLOSED_EYE)) == pytest.approx(0.1, abs=1e-3)

    def test_scale_invariance(self):
        small = eye_aspect_ratio(make_eye(OPEN_EYE))
        large = eye_aspect_ratio(make_eye(OPEN_EYE) * 7.0)
        assert small == pytest.approx(large, abs=1e-4)

    def test_depth_component_is_ignored(self):
        eye_2d = make_eye(OPEN_EYE)
        eye_3d = np.hstack([eye_2d, np.full((6, 1), 42.0, dtype=np.float32)])
        assert eye_aspect_ratio(eye_3d) == pytest.approx(eye_aspect_ratio(eye_2d), abs=1e-6)

    def test_wrong_point_count_raises(self):
        with pytest.raises(ValueError, match='6 eye points'):
            eye_aspect_ratio(np.zeros((5, 2), dtype=np.float32))


class TestComputeEyeAspectRatios:
    def test_returns_both_eyes(self):
        left, right = compute_eye_aspect_ratios(make_mesh(OPEN_EYE))
        assert left == pytest.approx(0.6, abs=1e-3)
        assert right == pytest.approx(0.6, abs=1e-3)

    def test_accepts_468_point_mesh(self):
        mesh = make_mesh(OPEN_EYE)[:468]
        left, right = compute_eye_aspect_ratios(mesh)
        assert left > 0 and right > 0

    def test_too_few_points_raise(self):
        with pytest.raises(ValueError, match='dense mesh'):
            compute_eye_aspect_ratios(np.zeros((106, 2), dtype=np.float32))


class TestBlinkDetector:
    def test_blink_is_counted_on_reopen(self):
        detector = BlinkDetector(min_consecutive_frames=2)
        frames = [OPEN_EYE, CLOSED_EYE, CLOSED_EYE, OPEN_EYE]
        events = [detector.update(make_mesh(opening)) for opening in frames]
        assert events == [False, False, False, True]
        assert detector.blink_count == 1

    def test_short_closure_is_jitter_not_blink(self):
        detector = BlinkDetector(min_consecutive_frames=2)
        events = [detector.update(make_mesh(opening)) for opening in [OPEN_EYE, CLOSED_EYE, OPEN_EYE]]
        assert events == [False, False, False]
        assert detector.blink_count == 0

    def test_multiple_blinks(self):
        detector = BlinkDetector(min_consecutive_frames=1)
        sequence = [OPEN_EYE, CLOSED_EYE, OPEN_EYE, CLOSED_EYE, CLOSED_EYE, OPEN_EYE]
        for opening in sequence:
            detector.update(make_mesh(opening))
        assert detector.blink_count == 2

    def test_eyes_closed_state(self):
        detector = BlinkDetector()
        detector.update(make_mesh(OPEN_EYE))
        assert detector.eyes_closed is False
        detector.update(make_mesh(CLOSED_EYE))
        assert detector.eyes_closed is True

    def test_reset(self):
        detector = BlinkDetector(min_consecutive_frames=1)
        for opening in [CLOSED_EYE, OPEN_EYE]:
            detector.update(make_mesh(opening))
        assert detector.blink_count == 1
        detector.reset()
        assert detector.blink_count == 0
        assert detector.eyes_closed is False

    def test_invalid_min_frames_raises(self):
        with pytest.raises(ValueError, match='min_consecutive_frames'):
            BlinkDetector(min_consecutive_frames=0)

    def test_repr(self):
        detector = BlinkDetector()
        assert 'BlinkDetector' in repr(detector)

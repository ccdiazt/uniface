# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Active-liveness helpers built on dense FaceMesh landmarks.

Passive anti-spoofing (`MiniFASNet`) judges a single frame; active liveness
asks the subject to do something a photo cannot, with blinking as the
lowest-friction challenge. Blink detection needs per-eye contours that
5-point detector landmarks cannot provide — these helpers compute the eye
aspect ratio (EAR) from a 468/478-point `FaceMesh` result and track it
across frames to count blinks.

The EAR is the ratio of the eye's vertical openings to its horizontal
width (Soukupova & Cech, "Real-Time Eye Blink Detection using Facial
Landmarks", CVWW 2016). Being a ratio, it is invariant to face scale and
distance to the camera. It hovers around 0.2-0.35 for an open eye and
collapses toward 0 when the eye closes.

Example:
    >>> from uniface import SCRFD, FaceMesh, BlinkDetector
    >>> detector, mesher, blinks = SCRFD(), FaceMesh(), BlinkDetector()
    >>> for frame in video_frames:
    ...     faces = detector.detect(frame)
    ...     if not faces:
    ...         continue
    ...     mesh = mesher.predict(frame, faces[:1])[0]
    ...     if blinks.update(mesh.landmarks):
    ...         print(f'Blink #{blinks.blink_count}')
"""

from __future__ import annotations

import numpy as np

from uniface.landmark._tessellation import NUM_MESH_LANDMARKS

__all__ = [
    'LEFT_EYE_EAR_INDICES',
    'RIGHT_EYE_EAR_INDICES',
    'BlinkDetector',
    'compute_eye_aspect_ratios',
    'eye_aspect_ratio',
]

#: Left-eye mesh indices for the EAR, ordered (inner corner, top-inner,
#: top-outer, outer corner, bottom-outer, bottom-inner).
LEFT_EYE_EAR_INDICES: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

#: Right-eye mesh indices for the EAR, same ordering as `LEFT_EYE_EAR_INDICES`.
RIGHT_EYE_EAR_INDICES: tuple[int, ...] = (33, 160, 158, 133, 153, 144)


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """Compute the eye aspect ratio from 6 eye contour points.

    Args:
        eye_points: (6, 2) or (6, 3) array ordered (corner, top, top, corner,
            bottom, bottom) — i.e. points 1 and 5, and 2 and 4, are vertical
            pairs while 0 and 3 are the horizontal corners. Any depth
            component is ignored.

    Returns:
        The EAR: (|p1-p5| + |p2-p4|) / (2 * |p0-p3|).

    Raises:
        ValueError: If `eye_points` does not contain exactly 6 points.
    """
    points = np.asarray(eye_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] != 6:
        raise ValueError(f'eye_aspect_ratio requires 6 eye points, got shape {points.shape}')
    points = points[:, :2]

    vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    return float(vertical / (2.0 * horizontal + 1e-6))


def compute_eye_aspect_ratios(landmarks: np.ndarray) -> tuple[float, float]:
    """Compute the per-eye aspect ratios from a dense FaceMesh result.

    Args:
        landmarks: (468, 2/3) or (478, 2/3) array from `FaceMesh`, as found in
            `FaceMeshResult.landmarks`. Both model variants share the first
            468 points, so either works.

    Returns:
        `(left_ear, right_ear)` — left and right refer to the subject's eyes.

    Raises:
        ValueError: If `landmarks` has fewer than 468 points.
    """
    points = np.asarray(landmarks)
    if points.ndim != 2 or points.shape[0] < NUM_MESH_LANDMARKS:
        raise ValueError(
            f'compute_eye_aspect_ratios requires a ({NUM_MESH_LANDMARKS}+, 2/3) dense mesh, got shape {points.shape}'
        )
    left = eye_aspect_ratio(points[list(LEFT_EYE_EAR_INDICES)])
    right = eye_aspect_ratio(points[list(RIGHT_EYE_EAR_INDICES)])
    return left, right


class BlinkDetector:
    """Stateful blink counter over a stream of FaceMesh landmarks.

    Feed one dense mesh per frame via `update()`. A blink is registered when
    the mean EAR stays below `ear_threshold` for at least
    `min_consecutive_frames` frames and then recovers — the frame count
    filters out landmark jitter and half-closed eyes.

    Args:
        ear_threshold: EAR below which the eyes count as closed. 0.21 works
            for most subjects at webcam framerates; lower it for narrow eyes,
            raise it if blinks are being missed.
        min_consecutive_frames: Closed frames required before the reopening
            counts as a blink.

    Attributes:
        blink_count: Total blinks registered since construction or `reset()`.
        eyes_closed: Whether the eyes were closed on the last `update()`.
    """

    def __init__(self, *, ear_threshold: float = 0.21, min_consecutive_frames: int = 2) -> None:
        if min_consecutive_frames < 1:
            raise ValueError(f'min_consecutive_frames must be >= 1, got {min_consecutive_frames}')
        self.ear_threshold = ear_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.blink_count = 0
        self._closed_frames = 0

    @property
    def eyes_closed(self) -> bool:
        """Whether the eyes were below the EAR threshold on the last frame."""
        return self._closed_frames > 0

    def update(self, landmarks: np.ndarray) -> bool:
        """Process one frame's landmarks and report whether a blink completed.

        Args:
            landmarks: Dense mesh for the current frame, shape (468+, 2/3).

        Returns:
            True exactly once per blink, on the frame where the eyes reopen
            after a qualifying closure.
        """
        left, right = compute_eye_aspect_ratios(landmarks)
        ear = (left + right) / 2.0

        if ear < self.ear_threshold:
            self._closed_frames += 1
            return False

        blinked = self._closed_frames >= self.min_consecutive_frames
        self._closed_frames = 0
        if blinked:
            self.blink_count += 1
        return blinked

    def reset(self) -> None:
        """Clear the blink count and closure state (e.g. between subjects)."""
        self.blink_count = 0
        self._closed_frames = 0

    def __repr__(self) -> str:
        return (
            f'BlinkDetector(ear_threshold={self.ear_threshold}, '
            f'min_consecutive_frames={self.min_consecutive_frames}, blinks={self.blink_count})'
        )

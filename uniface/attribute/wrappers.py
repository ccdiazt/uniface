# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""Adapters that plug anti-spoofing and quality models into FaceAnalyzer.

`MiniFASNet.predict` takes `(image, bbox)` and `EDifFIQA.predict` takes
`(image, landmarks)`, so neither fits the `BaseAttribute.predict(image, face)`
contract that `FaceAnalyzer` orchestrates. These wrappers bridge that gap:
they extract what the wrapped model needs from the `Face`, run it, and write
the result back (`face.is_real`/`face.spoofing_confidence`, `face.quality`),
letting a single `analyze()` call produce detection, embedding, liveness,
and quality in one pass.
"""

from __future__ import annotations

import numpy as np

from uniface.attribute.base import BaseAttribute
from uniface.quality.base import BaseQualityEstimator
from uniface.spoofing.base import BaseSpoofer
from uniface.types import Face, QualityResult, SpoofingResult

__all__ = ['QualityPredictor', 'SpoofingPredictor']


class SpoofingPredictor(BaseAttribute):
    """Runs a face anti-spoofing model as a FaceAnalyzer predictor.

    Wraps any `BaseSpoofer` (defaults to `MiniFASNet`) so presentation-attack
    detection participates in the `FaceAnalyzer` pipeline, enriching each
    `Face` with `is_real` and `spoofing_confidence`.

    Note that passive PAD judges the presentation, not the identity: apply it
    to live captures (selfies, webcam frames), never to photos of documents,
    which are replays by definition.

    Args:
        spoofer: Anti-spoofing model to wrap. Defaults to `MiniFASNet()`.

    Example:
        >>> from uniface import FaceAnalyzer, SpoofingPredictor
        >>> analyzer = FaceAnalyzer(predictors=[SpoofingPredictor()])
        >>> face = analyzer.analyze(image)[0]
        >>> face.is_real, face.spoofing_confidence
        (True, 0.98)
    """

    def __init__(self, *, spoofer: BaseSpoofer | None = None) -> None:
        if spoofer is None:
            from uniface.spoofing import MiniFASNet

            spoofer = MiniFASNet()
        self.spoofer = spoofer

    def _initialize_model(self) -> None:
        """The wrapped spoofer owns its inference session; nothing to load here."""

    def preprocess(self, image: np.ndarray, bbox: list | np.ndarray) -> np.ndarray:
        """Delegate preprocessing to the wrapped spoofer."""
        return self.spoofer.preprocess(image, bbox)

    def postprocess(self, prediction: np.ndarray) -> SpoofingResult:
        """Delegate postprocessing to the wrapped spoofer."""
        return self.spoofer.postprocess(prediction)

    def predict(self, image: np.ndarray, face: Face) -> SpoofingResult:
        """Run anti-spoofing on `face.bbox` and enrich the Face in-place.

        Args:
            image: The full input image in BGR format.
            face: Detected face; `face.bbox` locates the region to check.

        Returns:
            `SpoofingResult` with the is_real flag and confidence score.
        """
        result = self.spoofer.predict(image, face.bbox)
        face.is_real = result.is_real
        face.spoofing_confidence = result.confidence
        return result


class QualityPredictor(BaseAttribute):
    """Runs a face image quality model as a FaceAnalyzer predictor.

    Wraps any `BaseQualityEstimator` (defaults to `EDifFIQA`) so quality
    scoring participates in the `FaceAnalyzer` pipeline, enriching each
    `Face` with `quality`. Typical uses: reject degraded captures before
    matching, or pick the best frame of a short burst for enrollment.

    Requires a detector with `supports_alignment` (SCRFD, RetinaFace,
    CenterFace, YOLOv5Face, YOLOv8Face): the wrapped estimator aligns with
    the 5-point landmarks stored on the Face.

    Args:
        estimator: Quality model to wrap. Defaults to `EDifFIQA()`.

    Example:
        >>> from uniface import FaceAnalyzer, QualityPredictor
        >>> analyzer = FaceAnalyzer(predictors=[QualityPredictor()])
        >>> face = analyzer.analyze(image)[0]
        >>> face.quality
        0.7231
    """

    def __init__(self, *, estimator: BaseQualityEstimator | None = None) -> None:
        if estimator is None:
            from uniface.quality import EDifFIQA

            estimator = EDifFIQA()
        self.estimator = estimator

    def _initialize_model(self) -> None:
        """The wrapped estimator owns its inference session; nothing to load here."""

    def preprocess(self, image: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """Delegate preprocessing to the wrapped estimator (expects an aligned crop)."""
        return self.estimator.preprocess(image)

    def postprocess(self, prediction: QualityResult) -> QualityResult:
        """Quality estimators postprocess internally; passed through as-is."""
        return prediction

    def predict(self, image: np.ndarray, face: Face) -> QualityResult:
        """Score quality from `face.landmarks` and enrich the Face in-place.

        Args:
            image: The full input image in BGR format.
            face: Detected face; `face.landmarks` (5, 2) drive the alignment.

        Returns:
            `QualityResult` with the predicted score.
        """
        result = self.estimator.predict(image, face.landmarks)
        face.quality = result.score
        return result

# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
import pytest

from uniface.analyzer import FaceAnalyzer
from uniface.attribute import QualityPredictor, SpoofingPredictor
from uniface.detection.base import BaseDetector
from uniface.quality.base import BaseQualityEstimator
from uniface.spoofing.base import BaseSpoofer
from uniface.types import Face, QualityResult, SpoofingResult


class StubSpoofer(BaseSpoofer):
    """Returns a fixed result and records the bbox it was called with."""

    def __init__(self, is_real: bool = True, confidence: float = 0.97) -> None:
        self.result = SpoofingResult(is_real=is_real, confidence=confidence)
        self.seen_bboxes: list[np.ndarray] = []

    def _initialize_model(self) -> None: ...

    def preprocess(self, image, bbox):
        return image

    def postprocess(self, outputs):
        return self.result

    def predict(self, image, bbox) -> SpoofingResult:
        self.seen_bboxes.append(np.asarray(bbox))
        return self.result


class StubQualityEstimator(BaseQualityEstimator):
    """Returns a fixed score and records the landmarks it was called with."""

    def __init__(self, score: float = 0.73) -> None:
        self.result = QualityResult(score=score)
        self.seen_landmarks: list[np.ndarray] = []

    def _initialize_model(self) -> None: ...

    def preprocess(self, aligned_face):
        return aligned_face

    def score_aligned(self, aligned_face) -> QualityResult:
        return self.result

    def predict(self, image, landmarks) -> QualityResult:
        self.seen_landmarks.append(np.asarray(landmarks))
        return self.result


class StubDetector(BaseDetector):
    """Returns one fixed face without any model or download."""

    supports_landmarks = True
    supports_alignment = True

    def detect(self, image, **kwargs) -> list[Face]:
        return [
            Face(
                bbox=np.array([10.0, 20.0, 110.0, 140.0]),
                confidence=0.9,
                landmarks=np.arange(10, dtype=np.float32).reshape(5, 2),
            )
        ]

    def preprocess(self, image):
        return image

    def postprocess(self, outputs, **kwargs):
        return outputs


@pytest.fixture
def image():
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def face():
    return Face(
        bbox=np.array([10.0, 20.0, 110.0, 140.0]),
        confidence=0.9,
        landmarks=np.arange(10, dtype=np.float32).reshape(5, 2),
    )


class TestSpoofingPredictor:
    def test_enriches_face_in_place(self, image, face):
        predictor = SpoofingPredictor(spoofer=StubSpoofer(is_real=True, confidence=0.97))
        result = predictor.predict(image, face)

        assert isinstance(result, SpoofingResult)
        assert face.is_real is True
        assert face.spoofing_confidence == 0.97

    def test_uses_face_bbox(self, image, face):
        spoofer = StubSpoofer()
        SpoofingPredictor(spoofer=spoofer).predict(image, face)
        np.testing.assert_array_equal(spoofer.seen_bboxes[0], face.bbox)

    def test_fake_face(self, image, face):
        predictor = SpoofingPredictor(spoofer=StubSpoofer(is_real=False, confidence=0.88))
        predictor.predict(image, face)
        assert face.is_real is False
        assert face.spoofing_confidence == 0.88

    def test_callable_shortcut(self, image, face):
        predictor = SpoofingPredictor(spoofer=StubSpoofer())
        assert predictor(image, face) == predictor.predict(image, face)


class TestQualityPredictor:
    def test_enriches_face_in_place(self, image, face):
        predictor = QualityPredictor(estimator=StubQualityEstimator(score=0.73))
        result = predictor.predict(image, face)

        assert isinstance(result, QualityResult)
        assert face.quality == 0.73

    def test_uses_face_landmarks(self, image, face):
        estimator = StubQualityEstimator()
        QualityPredictor(estimator=estimator).predict(image, face)
        np.testing.assert_array_equal(estimator.seen_landmarks[0], face.landmarks)


class TestAnalyzerIntegration:
    def test_single_pass_produces_liveness_and_quality(self, image):
        """FaceAnalyzer orchestrates both adapters alongside detection."""
        analyzer = FaceAnalyzer(
            detector=StubDetector(),
            recognizer=None,
            predictors=[
                SpoofingPredictor(spoofer=StubSpoofer(is_real=True, confidence=0.95)),
                QualityPredictor(estimator=StubQualityEstimator(score=0.61)),
            ],
        )

        faces = analyzer.analyze(image)

        assert len(faces) == 1
        assert faces[0].is_real is True
        assert faces[0].spoofing_confidence == 0.95
        assert faces[0].quality == 0.61

    def test_repr_includes_spoofing_state(self, image):
        analyzer = FaceAnalyzer(
            detector=StubDetector(),
            recognizer=None,
            predictors=[SpoofingPredictor(spoofer=StubSpoofer(is_real=True, confidence=0.95))],
        )
        face = analyzer.analyze(image)[0]
        assert 'spoofing=Real(0.95)' in repr(face)

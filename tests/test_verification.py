# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
import pytest

from uniface.types import VerificationResult
from uniface.verification import DEFAULT_CONFIDENCE_BANDS, verify_faces


def unit_vector(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestVerifyFaces:
    def test_identical_embeddings_match_high(self):
        emb = unit_vector()
        result = verify_faces(emb, emb, normalized=True)
        assert result.is_match is True
        assert result.similarity == pytest.approx(1.0, abs=1e-5)
        assert result.confidence == 'high'

    def test_orthogonal_embeddings_do_not_match(self):
        emb1 = np.zeros(4, dtype=np.float32)
        emb2 = np.zeros(4, dtype=np.float32)
        emb1[0] = 1.0
        emb2[1] = 1.0
        result = verify_faces(emb1, emb2, normalized=True)
        assert result.is_match is False
        assert result.similarity == pytest.approx(0.0, abs=1e-6)
        assert result.confidence == 'no_match'

    def test_opposite_embeddings_yield_negative_similarity(self):
        emb = unit_vector()
        result = verify_faces(emb, -emb, normalized=True)
        assert result.is_match is False
        assert result.similarity == pytest.approx(-1.0, abs=1e-5)
        assert result.confidence == 'no_match'

    def test_unnormalized_inputs_are_normalized(self):
        emb = unit_vector()
        result = verify_faces(emb * 5.0, emb * 3.0)
        assert result.similarity == pytest.approx(1.0, abs=1e-4)

    def test_band_below_threshold_marks_review_zone(self):
        """A similarity in a named band but under the threshold is the manual-review case."""
        emb1 = np.array([1.0, 0.0], dtype=np.float32)
        # cos(x) = 0.45 with unit vectors
        angle = np.arccos(0.45)
        emb2 = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        result = verify_faces(emb1, emb2, threshold=0.5, normalized=True)
        assert result.is_match is False
        assert result.confidence == 'low'

    def test_similarity_at_threshold_matches(self):
        emb = unit_vector()
        result = verify_faces(emb, emb, threshold=1.0 - 1e-6, normalized=True)
        assert result.is_match is True

    def test_custom_bands(self):
        emb = unit_vector()
        result = verify_faces(emb, emb, bands={'strict': 0.7, 'lenient': 0.3}, normalized=True)
        assert result.confidence == 'strict'

    def test_empty_bands_raise(self):
        emb = unit_vector()
        with pytest.raises(ValueError, match='bands'):
            verify_faces(emb, emb, bands={}, normalized=True)

    def test_default_bands_are_ordered(self):
        cutoffs = list(DEFAULT_CONFIDENCE_BANDS.values())
        assert cutoffs == sorted(cutoffs, reverse=True)


class TestVerificationResult:
    def test_immutability(self):
        result = VerificationResult(is_match=True, similarity=0.8, confidence='high')
        with pytest.raises(AttributeError):
            result.is_match = False  # type: ignore

    def test_repr_match(self):
        result = VerificationResult(is_match=True, similarity=0.8123, confidence='high')
        repr_str = repr(result)
        assert 'Match' in repr_str
        assert '0.8123' in repr_str
        assert 'high' in repr_str

    def test_repr_no_match(self):
        result = VerificationResult(is_match=False, similarity=0.1, confidence='no_match')
        assert 'NoMatch' in repr(result)

    def test_hashable(self):
        result = VerificationResult(is_match=True, similarity=0.8, confidence='high')
        hash(result)

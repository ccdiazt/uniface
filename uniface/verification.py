# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""1:1 face verification with a decision threshold and confidence bands.

`compute_similarity` returns a raw cosine score and leaves the accept/reject
decision to the caller. This module adds that decision layer: a single call
that compares two embeddings, applies a threshold, and labels the score with
a confidence band so downstream systems (e.g. KYC pipelines) can route
borderline scores to manual review instead of forcing a binary outcome.
"""

from __future__ import annotations

import numpy as np

from uniface.face_utils import compute_similarity
from uniface.types import VerificationResult

__all__ = ['DEFAULT_CONFIDENCE_BANDS', 'verify_faces']

# Cutoffs follow the guidance in docs/concepts/thresholds-calibration.md:
# 0.4 low security, 0.5 balanced, 0.6 high security. They are a starting
# point, not a calibration — tune on your own genuine/impostor pairs.
DEFAULT_CONFIDENCE_BANDS: dict[str, float] = {'high': 0.6, 'medium': 0.5, 'low': 0.4}


def verify_faces(
    feat1: np.ndarray,
    feat2: np.ndarray,
    *,
    threshold: float = 0.4,
    bands: dict[str, float] | None = None,
    normalized: bool = False,
) -> VerificationResult:
    """Verify whether two face embeddings belong to the same person.

    Computes the cosine similarity between the two embeddings, decides the
    match against `threshold`, and labels the score with the highest
    confidence band whose cutoff it reaches.

    Bands are evaluated independently of the threshold. With
    `threshold=0.5` and the default bands, a similarity of 0.45 yields
    `is_match=False` but `confidence='low'` — a natural manual-review zone
    for identity-verification flows.

    Args:
        feat1: First embedding vector.
        feat2: Second embedding vector.
        threshold: Minimum cosine similarity to accept the pair as a match.
        bands: Mapping of band label to minimum similarity cutoff. The label
            with the highest cutoff not exceeding the similarity is reported;
            'no_match' is reported when the similarity is below every cutoff.
            Defaults to `DEFAULT_CONFIDENCE_BANDS`.
        normalized: Set True if the embeddings are already L2 normalized
            (e.g. from `get_normalized_embedding`), skipping renormalization.

    Returns:
        VerificationResult with the decision, raw similarity, and band label.

    Raises:
        ValueError: If `bands` is an empty mapping.

    Example:
        >>> from uniface import SCRFD, ArcFace, verify_faces
        >>> detector, recognizer = SCRFD(), ArcFace()
        >>> emb1 = recognizer.get_normalized_embedding(image1, detector.detect(image1)[0].landmarks)
        >>> emb2 = recognizer.get_normalized_embedding(image2, detector.detect(image2)[0].landmarks)
        >>> result = verify_faces(emb1, emb2, threshold=0.5, normalized=True)
        >>> result.is_match, result.confidence
        (True, 'high')
    """
    if bands is None:
        bands = DEFAULT_CONFIDENCE_BANDS
    if not bands:
        raise ValueError('bands must contain at least one label -> cutoff entry')

    similarity = float(compute_similarity(feat1, feat2, normalized=normalized))

    confidence = 'no_match'
    for label, cutoff in sorted(bands.items(), key=lambda item: item[1], reverse=True):
        if similarity >= cutoff:
            confidence = label
            break

    return VerificationResult(is_match=similarity >= threshold, similarity=similarity, confidence=confidence)

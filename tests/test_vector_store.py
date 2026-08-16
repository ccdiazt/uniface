# Copyright 2025-2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('faiss', reason='faiss-cpu is an optional dependency')

from uniface.stores import FAISS
from uniface.stores.base import BaseStore


def unit_vector(dim: int, hot: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[hot] = 1.0
    return vec


@pytest.fixture
def store(tmp_path):
    store = FAISS(embedding_size=8, db_path=str(tmp_path / 'index'))
    for i in range(4):
        store.add(unit_vector(8, i), {'person_id': str(i)})
    return store


class TestSearchTopK:
    def test_returns_ranked_matches(self, store):
        # Query correlated with vectors 0 and 1, stronger toward 0.
        query = unit_vector(8, 0) * 0.8 + unit_vector(8, 1) * 0.6
        results = store.search_topk(query, k=3, threshold=0.1)

        assert [meta['person_id'] for meta, _ in results] == ['0', '1']
        similarities = [sim for _, sim in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_threshold_filters_candidates(self, store):
        query = unit_vector(8, 0) * 0.8 + unit_vector(8, 1) * 0.6
        results = store.search_topk(query, k=3, threshold=0.7)
        assert [meta['person_id'] for meta, _ in results] == ['0']

    def test_k_caps_result_count(self, store):
        query = np.ones(8, dtype=np.float32) / np.sqrt(8)
        results = store.search_topk(query, k=2, threshold=0.0)
        assert len(results) == 2

    def test_k_larger_than_index(self, store):
        query = np.ones(8, dtype=np.float32) / np.sqrt(8)
        results = store.search_topk(query, k=100, threshold=0.0)
        assert len(results) == 4

    def test_empty_index_returns_empty_list(self, tmp_path):
        empty = FAISS(embedding_size=8, db_path=str(tmp_path / 'empty'))
        assert empty.search_topk(unit_vector(8, 0)) == []

    def test_no_candidate_clears_threshold(self, store):
        assert store.search_topk(unit_vector(8, 7), k=3, threshold=0.9) == []

    def test_invalid_k_raises(self, store):
        with pytest.raises(ValueError, match='k must be >= 1'):
            store.search_topk(unit_vector(8, 0), k=0)

    def test_base_store_default_raises(self):
        class MinimalStore(BaseStore):
            def add(self, embedding, metadata):
                pass

            def search(self, embedding, threshold=0.4):
                return None, 0.0

            def remove(self, key, value):
                return 0

            def __len__(self):
                return 0

        with pytest.raises(NotImplementedError, match='MinimalStore'):
            MinimalStore().search_topk(unit_vector(8, 0))


class TestSearchTopKConsistency:
    def test_top1_agrees_with_search(self, store):
        query = unit_vector(8, 2)
        best_meta, best_sim = store.search(query, threshold=0.4)
        topk = store.search_topk(query, k=1, threshold=0.4)

        assert best_meta is not None
        assert topk[0][0] == best_meta
        assert topk[0][1] == pytest.approx(best_sim)

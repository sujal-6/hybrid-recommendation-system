from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

Matrix = Union[csr_matrix, np.ndarray]

@dataclass
class ContentRecommender:
    # Content-based filtering using cosine similarity (spec).

    item_matrix: Matrix

    def recommend(
        self,
        query_vector: Matrix,
        *,
        k: int = 10,
        exclude: Optional[Set[int]] = None,
    ) -> Tuple[list[int], list[float]]:
        exclude = exclude or set()

        sims = cosine_similarity(query_vector, self.item_matrix).ravel()
        if exclude:
            sims[list(exclude)] = -np.inf

        k = min(k, sims.shape[0])
        if k <= 0:
            return [], []

        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return top_idx.tolist(), sims[top_idx].astype(float).tolist()
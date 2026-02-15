from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


@dataclass
class ItemItemCollaborativeRecommender:
    """
    Item-based collaborative filtering for implicit feedback (spec).

    Training builds normalized item profiles from the user-item matrix R:
      item_profiles = normalize(R.T)

    Scoring for a user sums cosine similarities from the user's interacted items
    to all items (optionally weighted by interaction strength).
    """

    item_profiles: Optional[csr_matrix] = None

    def fit(self, R: csr_matrix) -> "ItemItemCollaborativeRecommender":
        # item_profiles: shape (n_items, n_users), L2-normalized rows
        self.item_profiles = normalize(R.T.tocsr(), norm="l2", axis=1)
        return self

    def recommend(
        self,
        user_id: int,
        R: csr_matrix,
        *,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Tuple[list[int], list[float]]:
        if self.item_profiles is None:
            raise RuntimeError("Collaborative model not fitted")

        if user_id < 0 or user_id >= R.shape[0]:
            return [], []

        row = R.getrow(user_id)
        history_items = row.indices
        history_w = row.data.astype("float32")

        if history_items.size == 0:
            return [], []

        # Similarities from each history item to all items: (lenH, n_items)
        sims = self.item_profiles[history_items].dot(self.item_profiles.T)  # sparse

        # Weight each history row by interaction strength, then sum to a single score vector
        if history_w.size:
            sims = sims.multiply(history_w.reshape(-1, 1))
        scores = np.asarray(sims.sum(axis=0)).ravel().astype("float32")

        if exclude_seen:
            scores[history_items] = -np.inf

        k = min(k, scores.shape[0])
        if k <= 0:
            return [], []

        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx.tolist(), scores[top_idx].astype(float).tolist()

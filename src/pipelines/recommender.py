from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from src.core.hybrid import weighted_hybrid
from src.data.matrices import InteractionMappings

@dataclass
class Recommendation:
    item_idx: int
    score: float
    reason: str

class HybridRecommender:
    # Hybrid recommender:
    def __init__(
        self,
        *,
        content_model,
        collab_model,
        vectorizer,
        item_matrix,
        R: csr_matrix,
        mappings: InteractionMappings,
        users_by_id: Dict[int, Dict[str, Any]],
        opportunities_by_idx: List[Dict[str, Any]],
    ) -> None:
        self.content = content_model
        self.collab = collab_model
        self.vectorizer = vectorizer
        self.item_matrix = item_matrix
        self.R = R
        self.mappings = mappings
        self.users_by_id = users_by_id
        self.opportunities_by_idx = opportunities_by_idx

    def _seen_item_indices(self, user_row: int) -> set[int]:
        row = self.R.getrow(user_row)
        return set(row.indices.tolist())

    def _content_query_vector(self, *, query: Optional[str], user_row: Optional[int], user_id: Optional[int]):
        if query and query.strip():
            return self.vectorizer.transform([query])

        if user_id is not None:
            u = self.users_by_id.get(int(user_id))
            if u:
                profile_text = (u.get("profile_text") or "").strip()
                if profile_text:
                    return self.vectorizer.transform([profile_text])

        if user_row is not None and user_row >= 0:
            row = self.R.getrow(user_row)
            if row.nnz > 0:
                idx = row.indices
                w = row.data.astype("float32")

                if hasattr(self.item_matrix, "tocsr"):
                    mat = self.item_matrix[idx]
                    # weighted sum -> (1, dim)
                    centroid = mat.multiply(w.reshape(-1, 1)).sum(axis=0)
                    return centroid

                mat = np.asarray(self.item_matrix)[idx]
                centroid = (mat * w.reshape(-1, 1)).mean(axis=0, keepdims=True)
                return centroid

        return None

    def recommend(
        self,
        *,
        user_id: Optional[int] = None,
        query: Optional[str] = None,
        k: int = 10,
        alpha: float = 0.5,
    ) -> List[Recommendation]:
        user_row = None
        if user_id is not None:
            user_row = self.mappings.user_id_to_row.get(int(user_id))

        seen = set()
        if user_row is not None:
            seen = self._seen_item_indices(user_row)

        # Content
        c_dict: Dict[int, float] = {}
        qv = self._content_query_vector(query=query, user_row=user_row, user_id=user_id)
        if qv is not None:
            c_idx, c_sc = self.content.recommend(qv, k=k * 4, exclude=seen)
            c_dict = dict(zip(c_idx, c_sc))

        # Collaborative
        cf_dict: Dict[int, float] = {}
        if user_row is not None and self.R.shape[0] > 0:
            idx, sc = self.collab.recommend(user_row, self.R, k=k * 4, exclude_seen=True)
            cf_dict = dict(zip(idx, sc))

        # Fallback
        if not c_dict and not cf_dict:
            if self.R.nnz == 0:
                return []
            pop = np.asarray(self.R.sum(axis=0)).ravel()
            pop[list(seen)] = -np.inf
            kk = min(k, pop.shape[0])
            top = np.argpartition(-pop, kth=kk - 1)[:kk]
            top = top[np.argsort(-pop[top])]
            return [
                Recommendation(int(i), float(pop[i]), "Popular among users")
                for i in top.tolist()
                if np.isfinite(pop[i])
            ]

        ranked = weighted_hybrid(c_dict, cf_dict, alpha)
        out: List[Recommendation] = []
        for item_idx, score in ranked[:k]:
            reason = "Hybrid"
            if item_idx in c_dict and item_idx not in cf_dict:
                reason = "Content match (text similarity)"
            elif item_idx in cf_dict and item_idx not in c_dict:
                reason = "Collaborative match (users like you)"
            out.append(Recommendation(int(item_idx), float(score), reason))
        return out
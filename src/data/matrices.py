from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class InteractionMappings:
    user_id_to_row: Dict[int, int]
    row_to_user_id: Dict[int, int]
    opportunity_id_to_col: Dict[int, int]
    col_to_opportunity_id: Dict[int, int]


def build_interaction_matrix(
    *,
    opportunities: pd.DataFrame,
    interactions: pd.DataFrame,
) -> Tuple[csr_matrix, InteractionMappings]:
    """
    Builds a CSR user-item matrix R from interactions.

    - rows: mapped user indices (0..n_users-1)
    - cols: opportunity indices based on `opportunities` ordering
    - data: interaction weight (implicit feedback)
    """
    opp_ids = opportunities["opportunity_id"].astype(int).tolist()
    opportunity_id_to_col = {oid: i for i, oid in enumerate(opp_ids)}
    col_to_opportunity_id = {i: oid for oid, i in opportunity_id_to_col.items()}

    if interactions.empty:
        mappings = InteractionMappings({}, {}, opportunity_id_to_col, col_to_opportunity_id)
        return csr_matrix((0, len(opp_ids))), mappings

    # keep only known opportunities
    interactions = interactions.copy()
    interactions["opportunity_id"] = interactions["opportunity_id"].astype(int)
    interactions = interactions[interactions["opportunity_id"].isin(opportunity_id_to_col)]

    user_ids = sorted(set(interactions["user_id"].astype(int).tolist()))
    user_id_to_row = {uid: i for i, uid in enumerate(user_ids)}
    row_to_user_id = {i: uid for uid, i in user_id_to_row.items()}

    rows = interactions["user_id"].astype(int).map(user_id_to_row).astype(int).to_numpy()
    cols = interactions["opportunity_id"].astype(int).map(opportunity_id_to_col).astype(int).to_numpy()
    data = interactions.get("weight", 1.0).astype(float).to_numpy()

    R = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(opp_ids)))
    mappings = InteractionMappings(user_id_to_row, row_to_user_id, opportunity_id_to_col, col_to_opportunity_id)
    return R, mappings


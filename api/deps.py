import json
import pandas as pd

from src.core.vectorizer import TfidfVectorizerModel
from src.core.content import ContentRecommender
from src.core.collaborative import ItemItemCollaborativeRecommender
from src.data.db import DEFAULT_DB_PATH, fetch_interactions, fetch_opportunities, fetch_users, seed_from_csv
from src.data.matrices import build_interaction_matrix
from src.pipelines.recommender import HybridRecommender

class ModelStore:
    opps = None
    vectorizer = None
    recommender = None
    db_path = DEFAULT_DB_PATH


def reset_models() -> None:
    ModelStore.opps = None
    ModelStore.vectorizer = None
    ModelStore.recommender = None

def load_models():
    if ModelStore.recommender:
        return ModelStore

    print("🔹 Loading models...")

    # Seed SQLite from existing CSVs if empty
    seed_from_csv(db_path=ModelStore.db_path)

    opps = fetch_opportunities(ModelStore.db_path)
    if opps.empty:
        raise RuntimeError("No opportunities found. Add opportunities to the database first.")

    # Ensure expected optional columns exist
    for col in ["skills_required_json", "category", "location", "opportunity_type"]:
        if col not in opps.columns:
            opps[col] = ""

    # Build unified text field for CBF
    def _skills_text(v: str) -> str:
        try:
            arr = json.loads(v) if v else []
            return " ".join(arr) if isinstance(arr, list) else str(arr)
        except Exception:
            return str(v or "")

    opps = opps.copy()
    opps["skills_text"] = opps["skills_required_json"].apply(_skills_text)
    opps["full_text"] = (
        opps["title"].astype(str)
        + " "
        + opps["description"].astype(str)
        + " "
        + opps["skills_text"].astype(str)
        + " "
        + opps["category"].astype(str)
        + " "
        + opps["location"].astype(str)
        + " "
        + opps["opportunity_type"].astype(str)
    ).fillna("")

    # Content model (TF-IDF + cosine)
    vectorizer = TfidfVectorizerModel()
    item_matrix = vectorizer.fit_transform(opps["full_text"].tolist())
    content_model = ContentRecommender(item_matrix)

    # Collaborative model (item-item implicit)
    users = fetch_users(ModelStore.db_path)
    interactions = fetch_interactions(ModelStore.db_path)
    R, mappings = build_interaction_matrix(opportunities=opps, interactions=interactions)
    collab_model = ItemItemCollaborativeRecommender().fit(R) if R.shape[0] > 0 else ItemItemCollaborativeRecommender()

    # Simple user cache for cold-start profile-based recommendations
    users_by_id = {}
    if not users.empty:
        for _, r in users.iterrows():
            users_by_id[int(r["user_id"])] = {
                "profile_text": str(r.get("profile_text", "") or ""),
                "skills_json": str(r.get("skills_json", "") or "[]"),
                "interests_json": str(r.get("interests_json", "") or "[]"),
                "experience": str(r.get("experience", "") or ""),
            }

    opportunities_by_idx = opps.to_dict(orient="records")

    ModelStore.opps = opps
    ModelStore.vectorizer = vectorizer
    ModelStore.recommender = HybridRecommender(
        content_model=content_model,
        collab_model=collab_model,
        vectorizer=vectorizer,
        item_matrix=item_matrix,
        R=R,
        mappings=mappings,
        users_by_id=users_by_id,
        opportunities_by_idx=opportunities_by_idx,
    )

    print("✅ Models loaded")
    return ModelStore

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath("."))
from src.core.vectorizer import TfidfVectorizerModel
from src.core.content import ContentRecommender
from src.core.collaborative import ItemItemCollaborativeRecommender
from src.data.db import DEFAULT_DB_PATH, fetch_interactions, fetch_opportunities, fetch_users, seed_from_csv
from src.data.matrices import build_interaction_matrix
from src.pipelines.recommender import HybridRecommender
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg,
    mrr
)

def get_ground_truth(df_test):
    truth = {}
    for uid, group in df_test.groupby("user_id"):
        items = set(group["item_idx"].tolist())
        if items:
            truth[uid] = items
    return truth

def run_evaluation(K=5):
    print(" Starting Evaluation")
    seed_from_csv(db_path=DEFAULT_DB_PATH)
    opps = fetch_opportunities(DEFAULT_DB_PATH)
    opps = opps.fillna("")
    df = fetch_interactions(DEFAULT_DB_PATH)
    if df.empty:
        print(" No interactions found; cannot evaluate.")
        return

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    user_counts = train_df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    train_df = train_df[train_df["user_id"].isin(valid_users)]
    test_df = test_df[test_df["user_id"].isin(valid_users)]

    # CONTENT MODEL
    for col in ["skills_required_json", "category", "location", "opportunity_type"]:
        if col not in opps.columns:
            opps[col] = ""

    opps = opps.copy()
    opps["full_text"] = (
        opps["title"].astype(str)
        + " "
        + opps["description"].astype(str)
        + " "
        + opps["skills_required_json"].astype(str)
        + " "
        + opps["category"].astype(str)
        + " "
        + opps["location"].astype(str)
        + " "
        + opps["opportunity_type"].astype(str)
    ).fillna("")

    vectorizer = TfidfVectorizerModel()
    item_matrix = vectorizer.fit_transform(opps["full_text"].tolist())
    content_model = ContentRecommender(item_matrix)

    # COLLABORATIVE MODEL
    R_train, mappings = build_interaction_matrix(opportunities=opps, interactions=train_df)
    collab_model = ItemItemCollaborativeRecommender().fit(R_train) if R_train.shape[0] > 0 else ItemItemCollaborativeRecommender()

    users = fetch_users(DEFAULT_DB_PATH)
    users_by_id = {}
    if not users.empty:
        for _, r in users.iterrows():
            users_by_id[int(r["user_id"])] = {"profile_text": str(r.get("profile_text", "") or "")}

    system = HybridRecommender(
        content_model=content_model,
        collab_model=collab_model,
        vectorizer=vectorizer,
        item_matrix=item_matrix,
        R=R_train,
        mappings=mappings,
        users_by_id=users_by_id,
        opportunities_by_idx=opps.to_dict(orient="records"),
    )

    # EVALUATION
    test_df = test_df.copy()
    test_df["item_idx"] = test_df["opportunity_id"].astype(int).map(mappings.opportunity_id_to_col)
    test_df = test_df.dropna()
    test_df["item_idx"] = test_df["item_idx"].astype(int)

    ground_truth = get_ground_truth(test_df)

    metrics = {"p": [], "r": [], "ndcg": [], "mrr": []}

    for uid, relevant in ground_truth.items():
        recs = system.recommend(user_id=int(uid), query=None, k=K, alpha=0.5)
        rec_items = [r.item_idx for r in recs]

        metrics["p"].append(precision_at_k(rec_items, relevant, K))
        metrics["r"].append(recall_at_k(rec_items, relevant, K))
        metrics["ndcg"].append(ndcg(rec_items, relevant, K))
        metrics["mrr"].append(mrr(rec_items, relevant))

    print("\n" + "=" * 35)
    print(f" RESULTS @ K={K}")
    print("=" * 35)
    print(f"Precision@{K}: {np.mean(metrics['p']):.4f}")
    print(f"Recall@{K}:    {np.mean(metrics['r']):.4f}")
    print(f"NDCG@{K}:      {np.mean(metrics['ndcg']):.4f}")
    print(f"MRR:           {np.mean(metrics['mrr']):.4f}")


if __name__ == "__main__":
    run_evaluation(K=5)

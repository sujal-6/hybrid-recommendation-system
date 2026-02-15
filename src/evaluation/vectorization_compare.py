"""
Vectorization Comparison Script

Compares different vectorization techniques (TF-IDF, TF-IDF+SVD, SBERT)
on content-based recommendation performance.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("."))

from src.core.content import ContentRecommender
from src.core.vectorizer import SbertVectorizerModel, TfidfSvdVectorizerModel, TfidfVectorizerModel
from src.data.db import DEFAULT_DB_PATH, fetch_interactions, fetch_opportunities, seed_from_csv
from src.data.matrices import build_interaction_matrix
from src.evaluation.metrics import f1_score_at_k, ndcg, precision_at_k, recall_at_k


@dataclass
class VectorizationResult:
    name: str
    precision: float
    recall: float
    f1: float
    ndcg: float


def evaluate_vectorizer(
    vectorizer_name: str,
    vectorizer,
    opportunities: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    mappings,
    K: int = 5,
) -> VectorizationResult:
    """Evaluate a single vectorization technique"""
    print(f"  Evaluating {vectorizer_name}...")

    # Build text field
    opps = opportunities.copy()
    for col in ["skills_required_json", "category", "location", "opportunity_type"]:
        if col not in opps.columns:
            opps[col] = ""

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

    # Fit vectorizer
    item_matrix = vectorizer.fit_transform(opps["full_text"].tolist())
    content_model = ContentRecommender(item_matrix)

    # Build ground truth
    test_df = test_interactions.copy()
    test_df["item_idx"] = test_df["opportunity_id"].astype(int).map(mappings.opportunity_id_to_col)
    test_df = test_df.dropna()
    test_df["item_idx"] = test_df["item_idx"].astype(int)

    ground_truth = {}
    for uid, group in test_df.groupby("user_id"):
        items = set(group["item_idx"].tolist())
        if items:
            ground_truth[int(uid)] = items

    # Evaluate per user
    metrics = {"p": [], "r": [], "f1": [], "ndcg": []}

    for uid, relevant in ground_truth.items():
        # Build user profile from training history
        user_train = train_interactions[train_interactions["user_id"] == uid]
        if user_train.empty:
            continue

        user_train = user_train.copy()
        user_train["item_idx"] = user_train["opportunity_id"].astype(int).map(mappings.opportunity_id_to_col)
        user_train = user_train.dropna()
        if user_train.empty:
            continue

        history_idx = user_train["item_idx"].astype(int).tolist()
        if not history_idx:
            continue

        # Build query vector as mean of history items
        if hasattr(item_matrix, "tocsr"):
            # Sparse matrix
            mat = item_matrix[history_idx]
            query_vec = mat.mean(axis=0)
        else:
            # Dense matrix
            mat = np.asarray(item_matrix)[history_idx]
            query_vec = mat.mean(axis=0, keepdims=True)

        # Get recommendations
        rec_idx, _ = content_model.recommend(query_vec, k=K * 2)
        rec_items = rec_idx[:K]

        metrics["p"].append(precision_at_k(rec_items, relevant, K))
        metrics["r"].append(recall_at_k(rec_items, relevant, K))
        metrics["f1"].append(f1_score_at_k(rec_items, relevant, K))
        metrics["ndcg"].append(ndcg(rec_items, relevant, K))

    if not metrics["p"]:
        return VectorizationResult(vectorizer_name, 0.0, 0.0, 0.0, 0.0)

    return VectorizationResult(
        name=vectorizer_name,
        precision=np.mean(metrics["p"]),
        recall=np.mean(metrics["r"]),
        f1=np.mean(metrics["f1"]),
        ndcg=np.mean(metrics["ndcg"]),
    )


def generate_html_report(results: List[VectorizationResult], output_path: str, K: int = 5) -> None:
    """Generate HTML report for vectorization comparison"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Vectorization Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem; background-color: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 0.5rem; }}
        h2 {{ color: #555; margin-top: 2rem; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
        th, td {{ border: 1px solid #ddd; padding: 0.75rem; text-align: center; }}
        th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .metric {{ font-weight: bold; color: #2196F3; }}
        .summary {{ margin-top: 2rem; padding: 1rem; background-color: #e8f5e9; border-left: 4px solid #4CAF50; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Vectorization Comparison Report</h1>
        <p><strong>Evaluation Metric:</strong> K = {K}</p>
        <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Performance Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Vectorizer</th>
                    <th>Precision@K</th>
                    <th>Recall@K</th>
                    <th>F1-Score@K</th>
                    <th>NDCG@K</th>
                </tr>
            </thead>
            <tbody>
"""
    for r in results:
        html += f"""
                <tr>
                    <td><strong>{r.name}</strong></td>
                    <td class="metric">{r.precision:.4f}</td>
                    <td class="metric">{r.recall:.4f}</td>
                    <td class="metric">{r.f1:.4f}</td>
                    <td class="metric">{r.ndcg:.4f}</td>
                </tr>
"""

    # Find best performer
    best = max(results, key=lambda x: x.f1)
    html += f"""
            </tbody>
        </table>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Best Performing Vectorizer:</strong> {best.name} (F1-Score: {best.f1:.4f})</p>
            <p><strong>Analysis:</strong></p>
            <ul>
                <li><strong>TF-IDF:</strong> Fast, interpretable, good baseline for text similarity</li>
                <li><strong>TF-IDF + SVD:</strong> Dimensionality reduction can improve generalization on sparse data</li>
                <li><strong>SBERT:</strong> Semantic embeddings capture meaning better but require more computation</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Report saved to {output_path}")


def run_comparison(K: int = 5, output_path: Optional[str] = None) -> List[VectorizationResult]:
    """Run vectorization comparison"""
    print("🔹 Starting Vectorization Comparison")
    print(f"K = {K}")

    # Load data
    seed_from_csv(db_path=DEFAULT_DB_PATH)
    opps = fetch_opportunities(DEFAULT_DB_PATH)
    if opps.empty:
        print("❌ No opportunities found")
        return []

    interactions = fetch_interactions(DEFAULT_DB_PATH)
    if interactions.empty:
        print("❌ No interactions found")
        return []

    # Train/test split
    train_df, test_df = train_test_split(interactions, test_size=0.2, random_state=42)

    # Build mappings
    _, mappings = build_interaction_matrix(opportunities=opps, interactions=train_df)

    # Evaluate each vectorizer
    results = []

    # 1. TF-IDF (primary)
    print("\n1. TF-IDF Vectorizer")
    try:
        tfidf_vec = TfidfVectorizerModel()
        result = evaluate_vectorizer("TF-IDF", tfidf_vec, opps, train_df, test_df, mappings, K)
        results.append(result)
        print(f"   Precision@{K}: {result.precision:.4f}, Recall@{K}: {result.recall:.4f}, F1: {result.f1:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 2. TF-IDF + SVD
    print("\n2. TF-IDF + SVD Vectorizer")
    try:
        tfidf_svd_vec = TfidfSvdVectorizerModel(n_components=128)
        result = evaluate_vectorizer("TF-IDF+SVD", tfidf_svd_vec, opps, train_df, test_df, mappings, K)
        results.append(result)
        print(f"   Precision@{K}: {result.precision:.4f}, Recall@{K}: {result.recall:.4f}, F1: {result.f1:.4f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # 3. SBERT (optional, analysis only)
    print("\n3. SBERT Vectorizer (optional)")
    try:
        sbert_vec = SbertVectorizerModel()
        result = evaluate_vectorizer("SBERT", sbert_vec, opps, train_df, test_df, mappings, K)
        results.append(result)
        print(f"   Precision@{K}: {result.precision:.4f}, Recall@{K}: {result.recall:.4f}, F1: {result.f1:.4f}")
    except Exception as e:
        print(f"   ⚠️  SBERT not available (requires sentence-transformers): {e}")

    # Generate report
    if output_path is None:
        output_path = os.path.join("reports", "vectorization_report.html")

    if results:
        generate_html_report(results, output_path, K)
        print(f"\n✅ Comparison complete. Results:")
        for r in results:
            print(f"   {r.name}: F1={r.f1:.4f}, NDCG={r.ndcg:.4f}")

    return results


if __name__ == "__main__":
    run_comparison(K=5)

import numpy as np
from typing import List, Set, Union


def precision_at_k(ranked: List[int], relevant: Union[List[int], Set[int]], k: int) -> float:
    if k <= 0:
        return 0.0
    ranked_k = ranked[:k]
    relevant = set(relevant)
    return len(set(ranked_k) & relevant) / k


def recall_at_k(ranked: List[int], relevant: Union[List[int], Set[int]], k: int) -> float:
    relevant = set(relevant)
    if not relevant or k <= 0:
        return 0.0
    ranked_k = ranked[:k]
    return len(set(ranked_k) & relevant) / len(relevant)


def mrr(ranked: List[int], relevant: Union[List[int], Set[int]]) -> float:
    relevant = set(relevant)
    for i, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision_at_k(
    ranked: List[int],
    relevant: Union[List[int], Set[int]],
    k: int
) -> float:
    relevant = set(relevant)
    if not relevant or k <= 0:
        return 0.0

    score = 0.0
    hits = 0

    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(relevant), k)


def ndcg(ranked: List[int], relevant: Union[List[int], Set[int]], k: int) -> float:
    relevant = set(relevant)
    if not relevant or k <= 0:
        return 0.0

    ranked_k = ranked[:k]

    dcg = sum(
        (1.0 / np.log2(i + 2))
        for i, item in enumerate(ranked_k)
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def f1_score_at_k(ranked: List[int], relevant: Union[List[int], Set[int]], k: int) -> float:
    """F1-score at K: harmonic mean of precision@K and recall@K"""
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def rmse(predicted: List[float], actual: List[float]) -> float:
    """Root Mean Squared Error for rating predictions"""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have same length")
    if not predicted:
        return 0.0
    return float(np.sqrt(np.mean((np.array(predicted) - np.array(actual)) ** 2)))


def mae(predicted: List[float], actual: List[float]) -> float:
    """Mean Absolute Error for rating predictions"""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have same length")
    if not predicted:
        return 0.0
    return float(np.mean(np.abs(np.array(predicted) - np.array(actual))))

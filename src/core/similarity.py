import numpy as np
from scipy.spatial.distance import cdist


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between rows of A and rows of B
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return np.dot(A_norm, B_norm.T)


def euclidean_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Converts Euclidean distance into similarity
    """
    return -cdist(A, B, metric="euclidean")


def pearson_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pearson correlation similarity (row-wise)
    """
    A_centered = A - A.mean(axis=1, keepdims=True)
    B_centered = B - B.mean(axis=1, keepdims=True)

    return cosine_similarity(A_centered, B_centered)

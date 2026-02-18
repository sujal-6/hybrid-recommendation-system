from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def basic_preprocess(text: str) -> str:

    text = (text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class TextNormalizer(Protocol):
    def __call__(self, text: str) -> str: ...

class Vectorizer(ABC):
    @abstractmethod
    def fit_transform(self, corpus: Sequence[str]) -> Union[csr_matrix, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> Union[csr_matrix, np.ndarray]:
        raise NotImplementedError

class TfidfVectorizerModel(Vectorizer):
    # Primary vectorization for CBF (spec): TF-IDF.
    def __init__(
        self,
        *,
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        stop_words: str = "english",
        normalizer: TextNormalizer = basic_preprocess,
    ) -> None:
        self._normalizer = normalizer
        self._tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
            preprocessor=self._normalizer,
        )
        self._fitted = False

    def fit_transform(self, corpus: Sequence[str]) -> csr_matrix:
        X = self._tfidf.fit_transform(list(corpus))
        self._fitted = True
        return X.tocsr()

    def transform(self, texts: Sequence[str]) -> csr_matrix:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted")
        return self._tfidf.transform(list(texts)).tocsr()

    @property
    def feature_names(self) -> List[str]:
        return list(self._tfidf.get_feature_names_out())


class TfidfSvdVectorizerModel(Vectorizer):
    # Secondary (analysis): TF-IDF + TruncatedSVD.
    def __init__(
        self,
        *,
        max_features: int = 20000,
        n_components: int = 128,
        normalizer: TextNormalizer = basic_preprocess,
    ) -> None:
        self._tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            stop_words="english",
            preprocessor=normalizer,
        )
        self._svd = TruncatedSVD(n_components=n_components)
        self._fitted = False

    def fit_transform(self, corpus: Sequence[str]) -> np.ndarray:
        X = self._tfidf.fit_transform(list(corpus))
        n_samples, n_features = X.shape
        if n_samples <= 2 or n_features <= self._svd.n_components:
            vec = X.toarray().astype("float32")
        else:
            vec = self._svd.fit_transform(X).astype("float32")
        self._fitted = True
        return vec

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted")
        X = self._tfidf.transform(list(texts))
        n_samples, n_features = X.shape
        if n_samples <= 2 or n_features <= self._svd.n_components:
            return X.toarray().astype("float32")
        return self._svd.transform(X).astype("float32")


class SbertVectorizerModel(Vectorizer):
    # Secondary (analysis): SBERT embeddings.
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer 
            except Exception as e:
                raise RuntimeError(
                    "SBERT vectorizer requires `sentence-transformers` to be installed."
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit_transform(self, corpus: Sequence[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(list(corpus), normalize_embeddings=True).astype("float32")

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(list(texts), normalize_embeddings=True).astype("float32")
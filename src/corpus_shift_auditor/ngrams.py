"""Small, inspectable n-gram language model used by the shift auditor.

The production research track will add tuned interpolation and Kneser--Ney.
This additive-smoothed implementation is deliberately simple enough to audit
and already supports a useful end-to-end baseline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Iterable, Sequence


TOKEN_RE = re.compile(r"[\w]+(?:['’-][\w]+)*|[^\w\s]", flags=re.UNICODE)
START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"


def tokenize(text: str) -> list[str]:
    """Return a deterministic, lowercase word-and-punctuation tokenization."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass
class NGramLanguageModel:
    """Additive-smoothed word n-gram language model."""

    order: int = 3
    alpha: float = 0.1
    min_count: int = 1
    vocabulary: set[str] = field(default_factory=set)
    ngram_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)
    context_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be at least one")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.min_count < 1:
            raise ValueError("min_count must be at least one")

    def fit(self, texts: Iterable[str]) -> "NGramLanguageModel":
        """Fit vocabulary and n-gram counts from an iterable of documents."""

        documents = [tokenize(text) for text in texts]
        raw_counts = Counter(token for document in documents for token in document)
        self.vocabulary = {
            token for token, count in raw_counts.items() if count >= self.min_count
        }
        self.vocabulary.update({UNK, STOP})
        self.ngram_counts.clear()
        self.context_counts.clear()

        for document in documents:
            normalized = [token if token in self.vocabulary else UNK for token in document]
            for ngram in self._ngrams(normalized):
                self.ngram_counts[ngram] += 1
                self.context_counts[ngram[:-1]] += 1
        return self

    def _ngrams(self, tokens: Sequence[str]) -> list[tuple[str, ...]]:
        padding = [START] * (self.order - 1)
        sequence = padding + list(tokens) + [STOP]
        return [
            tuple(sequence[index : index + self.order])
            for index in range(len(sequence) - self.order + 1)
        ]

    def probability(self, ngram: Sequence[str]) -> float:
        """Return the smoothed conditional probability of one n-gram."""

        if not self.vocabulary:
            raise RuntimeError("fit the model before requesting probabilities")
        if len(ngram) != self.order:
            raise ValueError(f"expected an n-gram of length {self.order}")

        normalized = tuple(
            token if token in self.vocabulary or token == START else UNK for token in ngram
        )
        context = normalized[:-1]
        numerator = self.ngram_counts[normalized] + self.alpha
        denominator = self.context_counts[context] + self.alpha * len(self.vocabulary)
        return numerator / denominator

    def perplexity(self, texts: Iterable[str]) -> float:
        """Return token-weighted perplexity for one or more documents."""

        total_log_probability = 0.0
        predicted_tokens = 0
        for text in texts:
            normalized = [
                token if token in self.vocabulary else UNK for token in tokenize(text)
            ]
            for ngram in self._ngrams(normalized):
                total_log_probability += math.log(self.probability(ngram))
                predicted_tokens += 1
        if predicted_tokens == 0:
            raise ValueError("cannot compute perplexity for an empty corpus")
        return math.exp(-total_log_probability / predicted_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "alpha": self.alpha,
            "min_count": self.min_count,
            "vocabulary": sorted(self.vocabulary),
            "ngram_counts": [[list(key), value] for key, value in self.ngram_counts.items()],
            "context_counts": [
                [list(key), value] for key, value in self.context_counts.items()
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "NGramLanguageModel":
        model = cls(
            order=int(payload["order"]),
            alpha=float(payload["alpha"]),
            min_count=int(payload["min_count"]),
        )
        model.vocabulary = set(str(token) for token in payload["vocabulary"])
        model.ngram_counts = Counter(
            {tuple(key): int(value) for key, value in payload["ngram_counts"]}
        )
        model.context_counts = Counter(
            {tuple(key): int(value) for key, value in payload["context_counts"]}
        )
        return model


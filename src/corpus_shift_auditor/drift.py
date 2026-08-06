"""Corpus profiling and lexical/structural shift scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable

from .ngrams import NGramLanguageModel, tokenize


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class StructuralSummary:
    documents: int
    tokens: int
    sentences: int
    mean_sentence_tokens: float
    mean_token_characters: float
    punctuation_per_token: float


@dataclass
class CorpusProfile:
    """Persisted representation of the reference corpus."""

    name: str
    token_counts: dict[str, int]
    reference_perplexity: float
    structure: StructuralSummary
    language_model: NGramLanguageModel

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "name": self.name,
            "token_counts": self.token_counts,
            "reference_perplexity": self.reference_perplexity,
            "structure": asdict(self.structure),
            "language_model": self.language_model.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CorpusProfile":
        return cls(
            name=str(payload["name"]),
            token_counts={
                str(token): int(count) for token, count in payload["token_counts"].items()
            },
            reference_perplexity=float(payload["reference_perplexity"]),
            structure=StructuralSummary(**payload["structure"]),
            language_model=NGramLanguageModel.from_dict(payload["language_model"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CorpusProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass(frozen=True)
class AuditReport:
    reference_name: str
    risk_level: str
    risk_score: float
    perplexity: float
    perplexity_ratio: float
    out_of_vocabulary_rate: float
    lexical_js_divergence: float
    sentence_length_change: float
    token_length_change: float
    punctuation_rate_change: float
    incoming_structure: StructuralSummary

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["incoming_structure"] = asdict(self.incoming_structure)
        return result


def _materialize(texts: Iterable[str]) -> list[str]:
    documents = [text for text in texts if text.strip()]
    if not documents:
        raise ValueError("the corpus must contain at least one non-empty document")
    return documents


def summarize_structure(texts: Iterable[str]) -> StructuralSummary:
    documents = _materialize(texts)
    tokens_by_document = [tokenize(text) for text in documents]
    all_tokens = [token for document in tokens_by_document for token in document]
    word_tokens = [token for token in all_tokens if any(character.isalnum() for character in token)]
    sentences = [
        sentence
        for document in documents
        for sentence in SENTENCE_RE.split(document.strip())
        if sentence.strip()
    ]
    sentence_lengths = [len(tokenize(sentence)) for sentence in sentences]
    punctuation_count = sum(len(PUNCTUATION_RE.findall(text)) for text in documents)
    token_count = len(all_tokens)
    return StructuralSummary(
        documents=len(documents),
        tokens=token_count,
        sentences=len(sentences),
        mean_sentence_tokens=sum(sentence_lengths) / len(sentence_lengths),
        mean_token_characters=(
            sum(len(token) for token in word_tokens) / len(word_tokens) if word_tokens else 0.0
        ),
        punctuation_per_token=punctuation_count / max(token_count, 1),
    )


def build_profile(
    texts: Iterable[str],
    *,
    name: str = "reference",
    order: int = 3,
    alpha: float = 0.1,
    min_count: int = 1,
) -> CorpusProfile:
    documents = _materialize(texts)
    language_model = NGramLanguageModel(order=order, alpha=alpha, min_count=min_count)
    language_model.fit(documents)
    counts = Counter(token for document in documents for token in tokenize(document))
    return CorpusProfile(
        name=name,
        token_counts=dict(counts),
        reference_perplexity=language_model.perplexity(documents),
        structure=summarize_structure(documents),
        language_model=language_model,
    )


def _jensen_shannon(left: Counter[str], right: Counter[str]) -> float:
    support = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not support or left_total == 0 or right_total == 0:
        return 0.0

    divergence = 0.0
    for token in support:
        p = left[token] / left_total
        q = right[token] / right_total
        middle = 0.5 * (p + q)
        if p:
            divergence += 0.5 * p * math.log2(p / middle)
        if q:
            divergence += 0.5 * q * math.log2(q / middle)
    return divergence


def _relative_change(current: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if current == 0 else 1.0
    return abs(current - reference) / abs(reference)


def _risk_label(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "moderate"
    if score < 0.75:
        return "high"
    return "critical"


def audit_corpus(profile: CorpusProfile, texts: Iterable[str]) -> AuditReport:
    documents = _materialize(texts)
    incoming_tokens = [token for document in documents for token in tokenize(document)]
    incoming_counts = Counter(incoming_tokens)
    reference_counts = Counter(profile.token_counts)
    structure = summarize_structure(documents)

    perplexity = profile.language_model.perplexity(documents)
    perplexity_ratio = perplexity / max(profile.reference_perplexity, 1e-12)
    reference_vocabulary = set(profile.token_counts)
    oov_rate = (
        sum(token not in reference_vocabulary for token in incoming_tokens)
        / max(len(incoming_tokens), 1)
    )
    js_divergence = _jensen_shannon(reference_counts, incoming_counts)
    sentence_change = _relative_change(
        structure.mean_sentence_tokens, profile.structure.mean_sentence_tokens
    )
    token_change = _relative_change(
        structure.mean_token_characters, profile.structure.mean_token_characters
    )
    punctuation_change = _relative_change(
        structure.punctuation_per_token, profile.structure.punctuation_per_token
    )
    structural_score = min(1.0, (sentence_change + token_change + punctuation_change) / 3)
    perplexity_score = min(1.0, max(0.0, math.log2(max(perplexity_ratio, 1.0)) / 3))
    # Both quantities are naturally bounded in [0, 1]. Keeping that full
    # range avoids prematurely saturating modest shifts, which preserves the
    # distinction between partly familiar and genuinely out-of-domain text.
    lexical_score = js_divergence
    oov_score = oov_rate
    risk_score = (
        0.35 * perplexity_score
        + 0.25 * lexical_score
        + 0.20 * oov_score
        + 0.20 * structural_score
    )

    return AuditReport(
        reference_name=profile.name,
        risk_level=_risk_label(risk_score),
        risk_score=risk_score,
        perplexity=perplexity,
        perplexity_ratio=perplexity_ratio,
        out_of_vocabulary_rate=oov_rate,
        lexical_js_divergence=js_divergence,
        sentence_length_change=sentence_change,
        token_length_change=token_change,
        punctuation_rate_change=punctuation_change,
        incoming_structure=structure,
    )

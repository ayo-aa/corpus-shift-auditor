# Corpus Shift Auditor

Detect lexical and structural distribution shift before it quietly degrades an
NLP system.

```text
Reference: support tickets used to train a classifier
Incoming:  this week's production tickets

Lexical shift:          high
Out-of-vocabulary rate: 18.4%
Sentence-length shift:  moderate
Overall retraining risk: high
```

Corpus Shift Auditor builds an inspectable profile of a reference corpus and
then audits new text using n-gram perplexity, vocabulary change, lexical
distribution divergence, and structural summaries. The current `0.1` release
is a functional statistical baseline. The research track will add dependency
structure, neural representations, and tests of whether detected shift predicts
real downstream model degradation.

## Who this is for

- ML teams monitoring text-classification or extraction systems.
- Data teams comparing newly collected, synthetic, and historical corpora.
- Researchers studying when lexical and syntactic shift affect model quality.
- Teams that need a small, offline, interpretable alternative to an LLM-based auditor.

## Quick start

From the repository root:

```bash
python -m pip install -e .
corpus-shift-auditor fit examples/reference.txt \
  --model artifacts/reference.json \
  --name support-reference
corpus-shift-auditor audit examples/incoming_shifted.txt \
  --model artifacts/reference.json
```

The audit command emits machine-readable JSON containing the overall risk level
and each contributing measurement.

## Python API

```python
from corpus_shift_auditor import audit_corpus, build_profile

profile = build_profile(["Customers requested refunds for delayed orders."])
report = audit_corpus(profile, ["GPU workers failed during checkpoint recovery."])
print(report.risk_level, report.to_dict())
```

## What is implemented now

- Deterministic Unicode-aware tokenization.
- Additive-smoothed word n-gram model.
- Reference-corpus serialization.
- Perplexity ratio and out-of-vocabulary rate.
- Jensen--Shannon lexical divergence.
- Sentence-length, token-length, and punctuation shifts.
- CLI for fitting profiles and producing JSON audits.
- Unit and end-to-end CLI tests.

## Research question

> Do syntactic and representation-based signals detect consequential corpus
> shifts that lexical perplexity alone misses?

The planned experiment compares lexical, syntactic, embedding, and combined
detectors across chronological and domain shifts, then measures whether each
signal predicts degradation in downstream classifiers and structured predictors.
See [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Project status

This is the first conversion milestone: the repository is useful as a small
baseline and has a concrete research plan, but the dependency-parser and neural
experiments are not implemented yet. Results in `reports/` will only be promoted
to the README after reproducible multi-seed experiments are complete.

## Data and licensing

No course corpus is distributed here. Bring your own UTF-8 text or follow a
future licensed dataset adapter. See [DATA_USAGE.md](DATA_USAGE.md).

## Development

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check .
```

The code is available under the permissive [MIT License](LICENSE).

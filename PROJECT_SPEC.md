# Project specification

## Concrete problem

Production NLP systems are commonly evaluated on static held-out datasets, but
the language they receive changes over time. Teams need an inexpensive signal
that tells them when new data is sufficiently different to justify human review,
fresh evaluation, or retraining.

## Primary research question

Do syntactic and learned-representation signals identify performance-degrading
distribution shifts that are missed by lexical perplexity and vocabulary change?

## Hypotheses

1. Lexical n-gram signals will identify topic and vocabulary changes but miss
   structural changes expressed with familiar words.
2. Dependency-relation and dependency-distance distributions will improve
   detection of structural and genre shifts.
3. A combined detector will better predict downstream performance loss than any
   individual shift metric.
4. The best detector will depend on the downstream task; a universal scalar
   "drift score" will be less reliable than task-conditioned risk estimates.

## Users and decisions

The tool is intended to support three decisions:

- Is incoming data safe to process with the current model?
- Does the model need a fresh labeled evaluation?
- Which dimensions of the data changed enough to investigate?

The tool must never represent its heuristic risk score as a statistical guarantee.

## Experiment matrix

### Detectors

1. Vocabulary/OOV baseline.
2. N-gram perplexity and likelihood-ratio detector.
3. Lexical distribution divergence.
4. Dependency-relation frequency divergence.
5. Dependency-distance and tree-shape divergence.
6. Sentence-embedding distribution distance.
7. Learned supervised drift detector.
8. Combined calibrated detector.

### Shift types

- Topic/domain shift.
- Chronological shift.
- Genre and formality shift.
- Synthetic versus human text.
- Sentence-complexity shift.
- Controlled corruption and tokenization shift.
- Label-prevalence shift with similar input language.

### Downstream systems

- Text classifier.
- Token or span classifier.
- Dependency parser.
- Small language model.

### Primary outcomes

- Correlation between shift score and downstream performance loss.
- AUROC for detecting a predeclared unacceptable performance decline.
- Calibration of predicted performance risk.
- False-alarm rate at a fixed detection target.
- Runtime, memory, and corpus-size sensitivity.

## Statistical protocol

- Predeclare the primary outcome and unacceptable-degradation threshold.
- Use chronological or domain-held-out splits rather than random-only splits.
- Run at least three training seeds for learned downstream models.
- Bootstrap corpus-level confidence intervals.
- Select thresholds on validation domains and report final results on untouched domains.
- Include negative results and detector failure cases.

## Product milestones

### M0 — functional statistical baseline

- N-gram profile, lexical divergence, structural summary, CLI, tests.

### M1 — syntax-aware profile

- Open Universal Dependencies adapter.
- Dependency parser and relation/tree signatures.
- Parser confidence and failure handling.

### M2 — consequence benchmark

- Downstream tasks and controlled shifts.
- Performance-loss labels and detector comparison.

### M3 — operational tool

- Configurable thresholds.
- HTML/JSON reports.
- Batch and streaming modes.
- Historical trend view and monitoring integration.

### M4 — public research release

- Frozen data manifests.
- Multi-seed results.
- Reproduction commands.
- Report, model/data cards, and tagged release.

## Definition of done

- A new user can fit a reference profile and audit data in under ten minutes.
- Every published result is reproducible from a versioned configuration.
- The repository includes at least one syntactic and one learned detector.
- Evaluation connects shift scores to actual downstream degradation.
- Thresholds are validated on domains that differ from the final test domains.
- The README includes a real results table, limitations, and an operational example.


# Model card: statistical corpus-shift baseline

## Model description

The `0.1` baseline consists of an additive-smoothed word n-gram language model
plus corpus summary and divergence calculations. It creates a reference profile
and scores incoming text relative to that profile.

## Intended uses

- Early warning that production text differs from model-development text.
- Comparison of historical, incoming, or synthetic corpora.
- An inspectable baseline for distribution-shift research.

## Out-of-scope uses

- Automated denial of service or access to a person.
- Claims that a model has become unsafe solely because the risk score is high.
- Claims that a corpus is unchanged solely because the risk score is low.
- Language-quality judgments about individuals or demographic groups.

## Measurements

- Reference-model perplexity and perplexity ratio.
- Out-of-vocabulary rate.
- Jensen--Shannon divergence of token frequencies.
- Relative changes in sentence length, token length, and punctuation frequency.

## Limitations

- Tokenization is deliberately simple.
- The heuristic risk weights have not been empirically calibrated.
- Lexical metrics can miss syntactic, semantic, and label-distribution changes.
- Scores are not directly comparable across separately fitted profiles.
- Small incoming corpora produce noisy measurements.
- The baseline is not language-specific but has only been designed around
  whitespace-delimited text.

## Planned validation

Validation will measure whether each shift signal predicts downstream accuracy,
F1, calibration, or parsing degradation across unseen domains and chronological
splits. Until then, users should treat the output as an investigation prompt.


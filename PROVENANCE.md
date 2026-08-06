# Provenance and implementation boundaries

## Project lineage

The project grows from two completed Columbia NLP exercises:

- a trigram language model covering counts, unknown words, interpolated
  probabilities, perplexity, and corpus comparison;
- a transition-based dependency parser covering parser states, legal actions,
  learned action selection, and LAS/UAS evaluation.

Those artifacts established the technical foundation. This repository reframes
the work around a new concrete problem: detecting consequential text-corpus
shift. Its package structure, corpus-profile format, audit report, implementation,
tests, CLI, research design, and documentation were created for this project.

## Current implementation boundary

Version `0.1` contains a new additive-smoothed n-gram implementation and new
lexical/structural audit logic. It does not yet contain the dependency-parser
research track.

No original assignment prose, grading instructions, private URLs, supplied
test cases, trained checkpoints, or course datasets are distributed.

## Attribution expectations

Before public release, meaningful third-party code and AI-assisted contributions
must be recorded here or in commit/PR descriptions. External implementations
used for comparison should be cited but kept separate from the reference
implementation whenever possible.

## Claim boundaries

- The current tool is a statistical baseline, not a validated production monitor.
- Its risk score is a transparent heuristic, not a probabilistic guarantee.
- Future parser results must distinguish reimplementation from imported code.
- Results from prior course datasets must not be presented as results from this repository.


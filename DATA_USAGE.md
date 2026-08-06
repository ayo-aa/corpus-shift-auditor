# Data usage

## Current repository

The repository contains only small, newly written demonstration text files.
They are not intended to represent real users, companies, or production data.

The package accepts user-provided UTF-8 text. It does not upload text to an
external service.

## Data that is not included

- Brown corpus files from coursework.
- ETS/TOEFL essays.
- Supplied CoNLL/treebank files.
- Generated NumPy arrays or trained parser checkpoints.
- Personal or production text.

## Planned public datasets

The syntax research track should prefer Universal Dependencies treebanks with
licenses compatible with the intended use. Each selected treebank must be
recorded with:

- canonical source URL and version;
- checksum or immutable revision;
- license and attribution;
- preprocessing steps;
- train, validation, and test split policy.

Dataset licenses do not become the code license. Downloaded data should remain
outside Git and be ignored by default.

## Privacy

Real support, customer, or enterprise corpora may contain personal information.
Users are responsible for obtaining appropriate permission, minimizing data,
and applying retention and access controls. Future examples should use synthetic
or explicitly redistributable text.


"""Command-line interface for fitting and applying a corpus profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .drift import CorpusProfile, audit_corpus, build_profile


def _read_documents(paths: Sequence[str]) -> list[str]:
    return [Path(path).read_text(encoding="utf-8") for path in paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus-shift-auditor",
        description="Profile reference text and audit incoming corpora for distribution shift.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="Build a reference corpus profile")
    fit.add_argument("reference", nargs="+", help="One or more UTF-8 reference text files")
    fit.add_argument("--model", required=True, help="Destination JSON profile")
    fit.add_argument("--name", default="reference", help="Human-readable corpus name")
    fit.add_argument("--order", type=int, default=3, choices=(1, 2, 3, 4, 5))
    fit.add_argument("--alpha", type=float, default=0.1)
    fit.add_argument("--min-count", type=int, default=1)

    audit = commands.add_parser("audit", help="Audit incoming text against a profile")
    audit.add_argument("incoming", nargs="+", help="One or more UTF-8 incoming text files")
    audit.add_argument("--model", required=True, help="Reference JSON profile")
    audit.add_argument("--output", help="Optional destination for the JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        profile = build_profile(
            _read_documents(args.reference),
            name=args.name,
            order=args.order,
            alpha=args.alpha,
            min_count=args.min_count,
        )
        profile.save(args.model)
        print(f"Saved reference profile to {args.model}")
        return 0

    profile = CorpusProfile.load(args.model)
    report = audit_corpus(profile, _read_documents(args.incoming))
    rendered = json.dumps(report.to_dict(), indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved audit report to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


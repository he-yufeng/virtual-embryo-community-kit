"""Command-line entry point for VEC Data Inspector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .inspector import _attention_findings, inspect_h5ad, terminal_summary, write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read an .h5ad file without changing it and write reports to a separate directory."
    )
    parser.add_argument("input", type=Path, help="Path to an .h5ad file")
    parser.add_argument("--report-dir", type=Path, required=True, help="Separate directory for Markdown/JSON reports")
    parser.add_argument("--panel", type=Path, help="Optional newline-delimited gene panel to compare")
    parser.add_argument("--write-reorder-plan", action="store_true", help="Write a source-index reorder plan only when the panel has the same unique genes in a different order")
    parser.add_argument("--chunk-rows", type=int, default=1024, help="Dense/spatial rows per read block (default: 1024)")
    parser.add_argument("--chunk-items", type=int, default=1_000_000, help="Sparse/vector values per read block (default: 1000000)")
    parser.add_argument(
        "--fail-on-attention", action="store_true",
        help="Preserve reports but exit 1 if the reported attention section is nonempty; not a submission-validity check",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = inspect_h5ad(args.input, args.panel, chunk_rows=args.chunk_rows, chunk_items=args.chunk_items)
        outputs = write_reports(report, args.report_dir, write_reorder_plan=args.write_reorder_plan)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"vec-inspect: {error}", file=sys.stderr)
        return 2
    print(terminal_summary(report))
    print("  reports:")
    for label, path in outputs.items():
        print(f"    {label}: {path}")
    if args.fail_on_attention and _attention_findings(report):
        print("vec-inspect: attention findings detected; reports preserved. This is not a submission-validity decision.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

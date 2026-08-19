"""Insight Forge — CLI entry point.

Usage::

    python main.py <file.csv> [--crew] [--locale en] [--no-input]

Runs the full 8-stage pipeline end-to-end and prints a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Insight Forge — run the full analytics pipeline.",
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to the input CSV or XLSX file.",
    )
    parser.add_argument(
        "--crew",
        action="store_true",
        default=False,
        help="Run LLM agents through CrewAI (requires OPENROUTER_API_KEY in .env).",
    )
    parser.add_argument(
        "--locale",
        type=str,
        default="en",
        help="Report locale, e.g. 'en' or 'ar' (default: en).",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        default=False,
        help="Skip all interactive prompts; go straight to Generic Analysis Mode.",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.is_file():
        print(json.dumps({"error": f"File not found: {file_path}"}), file=sys.stderr)
        return 1

    if args.no_input:
        os.environ["INSIGHT_FORGE_NO_INPUT"] = "1"

    from crew.crew import run_pipeline

    try:
        result = run_pipeline(
            file_path=str(file_path),
            use_crew=args.crew,
            locale=args.locale,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    # Print summary
    summary = {
        "run_id": result.get("run_id"),
        "verdict": result.get("verdict"),
        "score": result.get("score"),
        "report_path": result.get("report_path"),
        "duration_s": round(result.get("duration_s", 0), 1),
    }
    print(json.dumps(summary, indent=2))

    # Exit code: 0 for approved, 1 for revision needed
    verdict = result.get("verdict", "NEEDS_REVISION")
    return 0 if verdict in ("APPROVED", "APPROVED_WITH_WARNINGS") else 1


if __name__ == "__main__":
    sys.exit(main())

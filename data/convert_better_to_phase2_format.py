#!/usr/bin/env python3
"""
Convert data/better.json into the same structure as phase2_advice_to10step_20251210.json.

Usage:
  python3 data/convert_better_to_phase2_format.py \
      --input data/better.json \
      --output data/better_phase2_format.json \
      --step 10
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple


def parse_entry(user_id: str, raw: Dict, step: int) -> Dict:
    """Parse one user entry from better.json into the phase2 format."""
    if not isinstance(raw, dict):
        raise ValueError("entry is not a dict")

    grade = raw.get("grade")
    raw_text = raw.get(user_id)
    if grade is None or raw_text is None:
        raise ValueError("missing grade or advice text")

    # Advice text is stored as 'assistant\\n\\n{...json...}'.
    match = re.search(r"({.*)", raw_text, flags=re.S)
    if not match:
        raise ValueError("could not locate inner JSON block")

    inner = json.loads(match.group(1))
    advice = inner.get(user_id)
    if not isinstance(advice, dict):
        raise ValueError("inner advice missing or not a dict")

    return {
        "userid": user_id,
        "grade": grade,
        "student_advice_title": advice.get("student_advice_title", ""),
        "student_advice_body": advice.get("student_advice_body", ""),
        "key_evidences": advice.get("key_evidences", []),
        "step": step,
    }


def convert(input_path: Path, output_path: Path, step: int) -> Tuple[int, int]:
    data = json.loads(input_path.read_text())

    converted = {}
    errors = []

    for user_id, raw in data.items():
        try:
            converted[user_id] = parse_entry(user_id, raw, step)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{user_id}: {exc}")

    output_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2))

    for err in errors:
        print(f"[warn] {err}", file=sys.stderr)

    return len(converted), len(errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert better.json into phase2-style JSON."
    )
    parser.add_argument(
        "--input", default="data/worse2.json", type=Path, help="path to json"
    )
    parser.add_argument(
        "--output",
        default="data/worse2_phase2_format.json",
        type=Path,
        help="output JSON path",
    )
    parser.add_argument(
        "--step",
        default=10,
        type=int,
        help="step value to set for each entry (default: 10)",
    )

    args = parser.parse_args()
    converted_count, error_count = convert(args.input, args.output, args.step)

    print(f"wrote {converted_count} entries to {args.output}")
    if error_count:
        print(f"{error_count} entries failed; see warnings above", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

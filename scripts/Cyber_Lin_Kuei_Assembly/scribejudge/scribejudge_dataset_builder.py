"""Build ScribeJudge training/audit rows from existing N05 handwritten maps.

This is not the final synthetic full-pipeline generator yet. It is the bridge:
run N05 however we want, then convert its full assembly output into stable rows
that can train a future referee model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
N05_DIR = ROOT / "scripts" / "N05handwritten_ocr"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(N05_DIR) not in sys.path:
    sys.path.insert(0, str(N05_DIR))

from scribejudge.dataset import build_scribejudge_rows, write_scribejudge_jsonl


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ScribeJudge JSONL rows from N05 maps.")
    parser.add_argument("--input", action="append", required=True, help="N05 *_handwritten_text_map.json path. Repeatable.")
    parser.add_argument("--output", default="datasets/scribejudge/scribejudge_audit_rows.jsonl")
    parser.add_argument("--settings", default="scripts/N05handwritten_ocr/settings.json")
    args = parser.parse_args()

    settings = load_json(ROOT / args.settings).get("scribejudge", {}) if args.settings else {}
    rows = []
    for input_path in args.input:
        payload = load_json(Path(input_path))
        assembly_map = payload.get("assembly") or {}
        document_rows = build_scribejudge_rows(
            assembly_map,
            settings=settings,
            truth_by_token_id={},
            base_dir=str(ROOT),
        )
        for row in document_rows:
            row["source_handwritten_text_map"] = str(Path(input_path))
            row["document_id"] = payload.get("document_id") or assembly_map.get("document_id")
        rows.extend(document_rows)

    output = write_scribejudge_jsonl(rows, ROOT / args.output)
    print("ScribeJudge rows:", len(rows))
    print("Output:", output)


if __name__ == "__main__":
    main()

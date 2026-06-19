"""JSON file helpers and command-line entrypoint for ScriLog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .parser import ScriLogParser

def parse_json_file(path: Path) -> Dict[str, Any]:
    """
    Read one JSON file.
    """

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    """
    Write one JSON file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def run_scrilog_on_payload(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run ScriLog on an already-loaded ScribeTrace payload.
    """

    parser = ScriLogParser()
    result = parser.parse_scribetrace_payload(payload)

    return result.to_dict()


def run_scrilog_on_file(
    input_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run ScriLog on one ScribeTrace JSON file.

    If output_path is provided:
        write result JSON there.

    Always returns:
        result dictionary.
    """

    payload = parse_json_file(input_path)

    result_dict = run_scrilog_on_payload(payload)

    if output_path is not None:
        write_json_file(
            path=output_path,
            payload=result_dict,
        )

    return result_dict


def build_cli_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.

    Example:

        python scrilog_monster.py input.json

        python scrilog_monster.py input.json --out output.json
    """

    cli = argparse.ArgumentParser(
        description=(
            "Run ScriLog Monster v0.1 on a ScribeTrace JSON payload "
            "after reconstruction/vector extraction."
        )
    )

    cli.add_argument(
        "input_json",
        type=str,
        help="Path to ScribeTrace JSON after reconstruction/vector extraction.",
    )

    cli.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path.",
    )

    return cli


def main() -> None:
    """
    CLI entrypoint.
    """

    cli = build_cli_parser()
    args = cli.parse_args()

    input_path = Path(args.input_json)

    output_path: Optional[Path] = None
    if args.out:
        output_path = Path(args.out)

    result_dict = run_scrilog_on_file(
        input_path=input_path,
        output_path=output_path,
    )

    if output_path is None:
        print(
            json.dumps(
                result_dict,
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
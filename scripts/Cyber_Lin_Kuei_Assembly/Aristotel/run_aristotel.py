"""Run the Aristotel degradation teacher from either script or package mode."""

import argparse
import json
import sys
from pathlib import Path


if __package__:
    from .recipes import build_default_recipes
    from .runner import AristotelRunner, FileCorrupter
else:
    # Direct execution does not create package context for relative imports.
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from scripts.Cyber_Lin_Kuei_Assembly.Aristotel.recipes import (
        build_default_recipes,
    )
    from scripts.Cyber_Lin_Kuei_Assembly.Aristotel.runner import (
        AristotelRunner,
        FileCorrupter,
    )


BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_ROOT = BASE_DIR / "Matenadata"
OUTPUT_ROOT = BASE_DIR / "datasets" / "aristotel_v0_1"


def build_argument_parser():
    """Create Aristotel's storage-safe command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic glyph degradations in memory or through "
            "an explicit storage mode."
        )
    )
    parser.add_argument(
        "--mode",
        choices=sorted(AristotelRunner.MODES),
        default="stream",
        help="stream stores nothing; export is the only full-image mode.",
    )
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument(
        "--recipe",
        action="append",
        dest="recipes",
        help="Run only this recipe; repeat the option for multiple recipes.",
    )
    return parser


def main(argv=None):
    """Run Aristotel and print only a compact JSON summary."""
    args = build_argument_parser().parse_args(argv)
    runner = AristotelRunner(
        input_root=args.input_root,
        output_root=args.output_root,
        corrupter=FileCorrupter(
            recipes=build_default_recipes(),
            seed=args.seed,
        ),
    )

    result = runner.run(
        mode=args.mode,
        limit=args.limit,
        epoch=args.epoch,
        variants=args.variants,
        recipe_names=args.recipes,
        preview_count=args.preview_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Backward-compatible ScriLog facade and direct CLI entrypoint."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from N05handwritten_ocr.scrilog.constants import *
    from N05handwritten_ocr.scrilog.engine import ScriLogEngine, ScriLogRule
    from N05handwritten_ocr.scrilog.facts import ScriLogFact, ScriLogFactBase, ScriLogFactKey
    from N05handwritten_ocr.scrilog.io import build_cli_parser, main, parse_json_file, run_scrilog_on_file, run_scrilog_on_payload, write_json_file
    from N05handwritten_ocr.scrilog.parser import ScriLogParser
    from N05handwritten_ocr.scrilog.profiles import ScriLogClassProfile, ScriLogProfileEvaluator
    from N05handwritten_ocr.scrilog.results import ScriLogCandidateEffect, ScriLogResult
    from N05handwritten_ocr.scrilog.rules import ScriLogRuleFactory
    from N05handwritten_ocr.scrilog.signature import ScriLogSignature
    from N05handwritten_ocr.scrilog.signature_builder import ScriLogSignatureBuilder
else:
    from .constants import *
    from .engine import ScriLogEngine, ScriLogRule
    from .facts import ScriLogFact, ScriLogFactBase, ScriLogFactKey
    from .io import build_cli_parser, main, parse_json_file, run_scrilog_on_file, run_scrilog_on_payload, write_json_file
    from .parser import ScriLogParser
    from .profiles import ScriLogClassProfile, ScriLogProfileEvaluator
    from .results import ScriLogCandidateEffect, ScriLogResult
    from .rules import ScriLogRuleFactory
    from .signature import ScriLogSignature
    from .signature_builder import ScriLogSignatureBuilder

__all__ = [
    "ScriLogCandidateEffect",
    "ScriLogClassProfile",
    "ScriLogEngine",
    "ScriLogFact",
    "ScriLogFactBase",
    "ScriLogFactKey",
    "ScriLogParser",
    "ScriLogProfileEvaluator",
    "ScriLogResult",
    "ScriLogRule",
    "ScriLogRuleFactory",
    "ScriLogSignature",
    "ScriLogSignatureBuilder",
    "build_cli_parser",
    "parse_json_file",
    "run_scrilog_on_file",
    "run_scrilog_on_payload",
    "write_json_file",
]

if __name__ == "__main__":
    main()

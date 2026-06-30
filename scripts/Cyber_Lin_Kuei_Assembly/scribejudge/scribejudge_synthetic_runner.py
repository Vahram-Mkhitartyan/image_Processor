"""Generate labeled ScribeJudge rows by running synthetic words through N05.

This is the first bridge from our synthetic word generator to the full N05
mixture-of-experts pipeline. It renders known words, wraps each render in a
minimal N03 visual-route contract, runs N05, then attaches truth labels to the
ScribeJudge rows.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
N05_DIR = ROOT / "scripts" / "N05handwritten_ocr"
ARENA_DIR = ROOT / "scripts" / "Cyber_Lin_Kuei_Assembly"
for path in (ROOT, N05_DIR, ARENA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from word_level_ocr_trainer import (  # noqa: E402
    build_tail_profiles,
    build_token_maps,
    collect_glyph_paths,
    load_json as load_arena_json,
    load_label_map,
    load_word_samples,
    render_synthetic_word,
)
from expert_orchestrator import build_handwriting_expert_map  # noqa: E402
from scribejudge.dataset import build_scribejudge_rows, write_scribejudge_jsonl  # noqa: E402


def save_json(data, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def render_visual_and_mask(rendered_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return visual black-on-white crop and ScribeTrace white-on-black mask."""

    if rendered_image.ndim != 2:
        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)
    _, visual = cv2.threshold(rendered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Generator visual convention is black ink on white. ScribeTrace masks use
    # white ink on black so topology tools see foreground as positive pixels.
    mask = cv2.bitwise_not(visual)
    return visual, mask


def write_synthetic_route(
    document_id: str,
    sample_dir: Path,
    visual_path: Path,
    mask_path: Path,
    image_shape: tuple[int, int],
) -> Path:
    """Write a minimal N03 route JSON accepted by N05."""

    height, width = image_shape
    bbox = {"x1": 0, "y1": 0, "x2": int(width), "y2": int(height)}
    route = {
        "document_id": document_id,
        "text_unit_id": 1,
        "group_id": "synthetic_word_0001",
        "source_group_id": "synthetic_word_0001",
        "source_layer_group_id": "synthetic_word_0001_blue",
        "layer": "blue",
        "source_crop_path": str(visual_path.resolve()),
        "refined_crop_path": str(visual_path.resolve()),
        "routed_crop_path": str(visual_path.resolve()),
        "original_crop_path": str(visual_path.resolve()),
        "analysis_crop_path": str(visual_path.resolve()),
        "classification_crop_path": str(visual_path.resolve()),
        "classification_crop_source": "scribejudge_synthetic_visual",
        "classification_crop_policy": "synthetic_single_word",
        "context_crop_path": str(visual_path.resolve()),
        "analysis_mask_crop_path": str(mask_path.resolve()),
        "mask_source": "blue_continuity_mask",
        "visual_layer_source": "blue_ink_layer",
        "bbox": bbox,
        "crop_bbox": bbox,
        "final_bbox": bbox,
        "layer_hypothesis": "blue_handwriting",
        "role_guess": "synthetic_word",
        "recommended_next_node": "N05_handwritten_ocr",
        "minos_required": False,
        "minos_mode": "not_required",
        "is_final_text_candidate": True,
        "preserve_as_evidence": False,
        "force_handwritten_ocr": True,
        "visual_classification": {
            "node": "scribejudge_synthetic_runner",
            "model": "synthetic_truth_route",
            "model_version": "0.1",
            "visual_class": "handwriting_only",
            "recommended_route": ["N05_handwritten_ocr"],
            "scores": {"handwriting_only": 1.0, "mixed": 0.0, "printed_only": 0.0},
            "thresholds": {},
        },
    }
    payload = {
        "document_id": document_id,
        "routes": [route],
        "summary": {
            "source": "scribejudge_synthetic_runner",
            "route_count": 1,
            "truth_known": True,
        },
    }
    route_path = sample_dir / "n03_visual_classification" / "metadata" / f"{document_id}_n03_visual_classification_routes.json"
    return save_json(payload, route_path)


def load_generation_assets(settings_path: Path):
    settings = load_arena_json(settings_path)
    label_map = load_label_map(settings["dataset"]["label_map_path"])
    char_to_token, token_to_char = build_token_maps(label_map)
    tail_profiles = build_tail_profiles(token_to_char)
    samples = load_word_samples(settings, char_to_token)
    glyph_paths = collect_glyph_paths(settings["dataset"]["matenadata_dir"], label_map)
    return settings, token_to_char, tail_profiles, samples, glyph_paths


def run_one_sample(
    index: int,
    word_sample,
    token_to_char: dict[int, str],
    tail_profiles: dict[int, str],
    glyph_paths: dict[int, list[Path]],
    generation_settings: dict,
    n05_settings_path: Path,
    run_dir: Path,
    seed: int,
) -> tuple[list[dict], dict]:
    document_id = f"sj_{index:05d}_{word_sample.text}"
    sample_dir = run_dir / document_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + index)
    rendered = render_synthetic_word(
        word_sample,
        glyph_paths=glyph_paths,
        tail_profiles=tail_profiles,
        rendering=generation_settings["rendering"],
        rng=rng,
    )
    visual, mask = render_visual_and_mask(rendered.image)
    crop_dir = sample_dir / "synthetic_source"
    crop_dir.mkdir(parents=True, exist_ok=True)
    visual_path = crop_dir / "word_visual.png"
    mask_path = crop_dir / "word_mask.png"
    cv2.imwrite(str(visual_path), visual)
    cv2.imwrite(str(mask_path), mask)
    route_path = write_synthetic_route(
        document_id=document_id,
        sample_dir=sample_dir,
        visual_path=visual_path,
        mask_path=mask_path,
        image_shape=visual.shape[:2],
    )
    n05_output_dir = sample_dir / "n05_handwritten_ocr"
    n05_result = build_handwriting_expert_map(
        visual_routes_path=str(route_path),
        output_dir=str(n05_output_dir),
        settings_path=str(n05_settings_path),
    )
    truth_tokens = [token_to_char[token_id] for token_id in word_sample.token_ids]
    truth_by_token_id = {
        "1": {
            "text": word_sample.text,
            "tokens": truth_tokens,
        }
    }
    rows = build_scribejudge_rows(
        n05_result.get("assembly") or {},
        settings=(n05_result.get("assembly") or {}).get("settings", {}).get("scribejudge", {}),
        truth_by_token_id=truth_by_token_id,
        base_dir=str(ROOT),
    )
    for row in rows:
        row["synthetic"] = {
            "document_id": document_id,
            "truth_text": word_sample.text,
            "truth_tokens": truth_tokens,
            "visual_path": str(visual_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "n05_metadata_path": n05_result.get("metadata_path"),
            "split_x_positions": list(rendered.split_x_positions),
            "bridge_count": rendered.bridge_count,
            "transition_count": rendered.transition_count,
        }
    summary = {
        "document_id": document_id,
        "truth_text": word_sample.text,
        "truth_tokens": truth_tokens,
        "row_count": len(rows),
        "selected_texts": [row.get("selected_text") for row in rows],
        "char_accuracy": rows[0].get("char_accuracy") if rows else None,
        "is_exact": rows[0].get("is_exact") if rows else None,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic words through N05 and label ScribeJudge rows.")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--word-settings", default="scripts/Cyber_Lin_Kuei_Assembly/word_level_ocr_settings.json")
    parser.add_argument("--n05-settings", default="scripts/N05handwritten_ocr/settings.json")
    parser.add_argument("--output", default="datasets/scribejudge/synthetic_scribejudge_rows.jsonl")
    parser.add_argument("--run-dir", default="temp_processing/scribejudge_synthetic")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--max-word-length", type=int, default=14)
    args = parser.parse_args()

    run_dir = ROOT / args.run_dir
    if run_dir.exists() and not args.keep_existing:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    generation_settings, token_to_char, tail_profiles, samples, glyph_paths = load_generation_assets(
        ROOT / args.word_settings
    )
    samples = [sample for sample in samples if len(sample.token_ids) <= args.max_word_length]
    rng = random.Random(args.seed)
    selected_samples = [rng.choice(samples) for _ in range(args.samples)]

    all_rows: list[dict] = []
    summaries = []
    started = time.time()
    for index, sample in enumerate(selected_samples, start=1):
        try:
            rows, summary = run_one_sample(
                index=index,
                word_sample=sample,
                token_to_char=token_to_char,
                tail_profiles=tail_profiles,
                glyph_paths=glyph_paths,
                generation_settings=generation_settings,
                n05_settings_path=ROOT / args.n05_settings,
                run_dir=run_dir,
                seed=args.seed,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            print(
                f"[{index}/{len(selected_samples)}] {sample.text} -> "
                f"{summary.get('selected_texts')} char_acc={summary.get('char_accuracy')} exact={summary.get('is_exact')}"
            )
        except Exception as error:  # Keep the batch alive; failed samples are training signal too.
            summaries.append({"index": index, "truth_text": sample.text, "error": str(error)})
            print(f"[{index}/{len(selected_samples)}] {sample.text} FAILED: {error}")

    output_path = write_scribejudge_jsonl(all_rows, ROOT / args.output)
    report = {
        "status": "completed",
        "sample_count": len(selected_samples),
        "row_count": len(all_rows),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_jsonl": str(output_path),
        "run_dir": str(run_dir),
        "exact_count": sum(1 for row in all_rows if row.get("is_exact") is True),
        "average_char_accuracy": (
            sum(float(row.get("char_accuracy") or 0.0) for row in all_rows)
            / max(len(all_rows), 1)
        ),
        "backup_recovery_opportunity_count": sum(
            int(row.get("backup_recovery_opportunity_count") or 0) for row in all_rows
        ),
        "truth_missing_from_candidates_count": sum(
            int(row.get("truth_missing_from_candidates_count") or 0) for row in all_rows
        ),
        "samples": summaries,
    }
    report_path = run_dir / "scribejudge_synthetic_report.json"
    save_json(report, report_path)
    print("ScribeJudge synthetic rows:", len(all_rows))
    print("Output:", output_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()

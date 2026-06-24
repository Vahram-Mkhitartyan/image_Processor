"""Build the coordinate-aware N05 handwriting mixture-of-experts map."""

import json
import os
import shutil

try:
    from .assembly import build_assembly_map
    from .character_unit_proposer import propose_character_units
    from .character_detector import get_expert_manifest as get_character_manifest
    from .scribetrace import get_expert_manifest as get_scribetrace_manifest
    from .scribetrace import recognize as recognize_scribetrace
    from .tesseract_ocr import get_expert_manifest as get_tesseract_manifest
    from .word_level_ocr import get_expert_manifest as get_word_level_manifest
    from .word_level_ocr import recognize as recognize_word_level
except ImportError:
    from assembly import build_assembly_map
    from character_unit_proposer import propose_character_units
    from character_detector import get_expert_manifest as get_character_manifest
    from scribetrace import get_expert_manifest as get_scribetrace_manifest
    from scribetrace import recognize as recognize_scribetrace
    from tesseract_ocr import get_expert_manifest as get_tesseract_manifest
    from word_level_ocr import get_expert_manifest as get_word_level_manifest
    from word_level_ocr import recognize as recognize_word_level


NODE_NAME = "N05_handwriting_expert_orchestrator"
NODE_VERSION = "0.3.1"
ORCHESTRATOR_NAME = "handwriting_mixture_of_experts"
HANDWRITING_VISUAL_CLASSES = {"handwriting_only", "mixed"}
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SETTINGS_PATH = os.path.join(MODULE_DIR, "settings.json")


def load_json(input_path):
    """Load a JSON file from disk."""
    with open(input_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, output_path):
    """Save Python data as readable JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return output_path


def ensure_dir(*paths):
    """Create directories and return the supplied paths."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
    return paths


def check_file_exists(path, label):
    """Fail early when a required path is missing."""
    if not path:
        raise FileNotFoundError(f"{label}: path is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_settings(settings_path=None):
    """Load optional N05 settings, defaulting to the node settings file."""
    path = settings_path or DEFAULT_SETTINGS_PATH
    if not path or not os.path.isfile(path):
        return {}
    return load_json(path)


def build_expert_registry(settings):
    """Build the four-expert registry without eagerly loading model weights."""
    expert_settings = settings.get("experts", {})
    return {
        "tesseract_ocr": get_tesseract_manifest(
            expert_settings.get("tesseract_ocr", {})
        ),
        "scribetrace": get_scribetrace_manifest(
            expert_settings.get("scribetrace", {})
        ),
        "character_detector": get_character_manifest(
            expert_settings.get("character_detector", {})
        ),
        "word_level_ocr": get_word_level_manifest(
            expert_settings.get("word_level_ocr", {})
        ),
    }


def create_output_folders(output_dir):
    """Create and return the debug-friendly N05 output structure."""
    folders = {
        "root": os.path.abspath(output_dir),
        "output_dir": os.path.abspath(output_dir),
        "crops": os.path.join(output_dir, "crops"),
        "handwriting_only": os.path.join(output_dir, "crops", "handwriting_only"),
        "mixed": os.path.join(output_dir, "crops", "mixed"),
        "fallback_from_printed_only": os.path.join(
            output_dir,
            "crops",
            "fallback_from_printed_only",
        ),
        "metadata": os.path.join(output_dir, "metadata"),
        "debug": os.path.join(output_dir, "debug"),
        "skeletons": os.path.join(output_dir, "debug", "skeletons"),
        "overlays": os.path.join(output_dir, "debug", "overlays"),
        "scribetrace": os.path.join(output_dir, "scribetrace"),
        "scribetrace_debug": os.path.join(output_dir, "scribetrace", "debug"),
        "character_unit_proposer": os.path.join(
            output_dir,
            "character_unit_proposer",
        ),
        "character_unit_segments": os.path.join(
            output_dir,
            "character_unit_proposer",
            "segments",
        ),
        "character_unit_debug": os.path.join(
            output_dir,
            "character_unit_proposer",
            "debug",
        ),
        "assembly": os.path.join(output_dir, "assembly"),
    }
    ensure_dir(
        folders["root"],
        folders["metadata"],
        folders["scribetrace"],
        folders["scribetrace_debug"],
        folders["assembly"],
    )
    return folders


def reset_output_dir(output_dir):
    """Delete previous N05 artifacts so reruns cannot retain stale segments."""
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def load_visual_routes(visual_routes_path):
    """Load and validate the N03 visual-route contract."""
    check_file_exists(visual_routes_path, "N03 visual routes JSON")
    payload = load_json(visual_routes_path)
    if "routes" not in payload:
        raise KeyError("N03 routes JSON has no 'routes' key.")
    return payload


def get_visual_class(route_record):
    """Extract the stable visual class from one N03 route."""
    return (
        route_record.get("visual_classification", {}).get("visual_class")
        or route_record.get("visual_class")
        or "review"
    )


def should_send_to_handwritten_ocr(route_record):
    """Return whether an N03 route belongs in N05."""
    return (
        route_record.get("force_handwritten_ocr") is True
        or get_visual_class(route_record) in HANDWRITING_VISUAL_CLASSES
    )


def select_handwriting_candidates(route_records):
    """Select N03 handwriting-only and mixed records deterministically."""
    return [
        route_record
        for route_record in route_records
        if should_send_to_handwritten_ocr(route_record)
    ]


def _normalized_bbox(bbox):
    """Return a JSON-safe bounding box with derived dimensions."""
    if not isinstance(bbox, dict):
        return None
    try:
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": max(0, x2 - x1),
        "height": max(0, y2 - y1),
    }


def build_document_bbox(route_record):
    """Build the preferred original-document coordinate box."""
    return _normalized_bbox(
        route_record.get("final_bbox")
        or route_record.get("bbox")
        or route_record.get("crop_bbox")
    )


def build_crop_bbox(route_record):
    """Build the available crop-coordinate box."""
    return _normalized_bbox(
        route_record.get("crop_bbox")
        or route_record.get("final_bbox")
        or route_record.get("bbox")
    )


def get_best_crop_path_with_source(route_record):
    """Return the best visual handwriting crop and its source field."""
    for key in (
        "classification_crop_path",
        "routed_crop_path",
        "analysis_crop_path",
        "refined_crop_path",
        "original_crop_path",
        "context_crop_path",
        "source_crop_path",
    ):
        path = route_record.get(key)
        if path and os.path.isfile(path):
            return os.path.abspath(path), key
    return None, None


def get_best_crop_path(route_record):
    """Compatibility helper returning only the selected visual crop path."""
    return get_best_crop_path_with_source(route_record)[0]


def copy_candidate_crop(route_record, folders, save_copy=False):
    """Select one N03 crop and optionally materialize an N05 debug copy."""
    crop_path, crop_source = get_best_crop_path_with_source(route_record)
    if crop_path is None:
        return None, None, None

    if not save_copy:
        return None, crop_path, crop_source

    visual_class = get_visual_class(route_record)
    target_folder = folders.get(visual_class, folders["fallback_from_printed_only"])
    ensure_dir(target_folder)
    target_path = os.path.join(target_folder, os.path.basename(crop_path))
    shutil.copy2(crop_path, target_path)
    return os.path.abspath(target_path), crop_path, crop_source


def build_placeholder_handwriting_result():
    """Build the non-attempted raw handwriting OCR result contract."""
    return {
        "attempted": False,
        "engine_family": None,
        "engine_version": None,
        "status": "pending_handwriting_engine",
        "trusted_as_final": False,
        "engines_used": [],
        "candidates": [],
        "error": None,
    }


def build_default_handwriting_profile():
    """Return neutral handwriting-profile evidence."""
    return {
        "style": "unknown",
        "connectedness": None,
        "curvature": None,
        "stroke_clarity": None,
        "letter_separation": None,
        "spacing_quality": None,
        "likely_unit_type": "unknown",
    }


def build_default_engine_routing():
    """Return neutral mixture-of-experts routing evidence."""
    return {
        "selected_strategy": "not_run",
        "reason": None,
        "tool_weights": {},
    }


def build_default_edge_cases():
    """Return explicit false defaults for handwriting edge-case flags."""
    return {
        key: False
        for key in (
            "broken_letter_possible",
            "connected_fragment",
            "overwritten_or_corrected",
            "crossed_out",
            "line_contact",
            "tiny_marks_nearby",
            "possible_abbreviation",
            "contains_digits",
            "signature_like",
            "block_handwriting",
            "mixed_language_possible",
            "short_text_risk",
            "shaky_strokes",
            "possible_empty_field",
        )
    }


def build_handwritten_text_unit(
    route_record,
    copied_crop_path,
    selected_crop_path,
    selected_crop_source,
    folders,
    document_id,
):
    """Build one coordinate-aware N05 handwritten text-unit record."""
    classification = route_record.get("visual_classification", {})
    return {
        "document_id": route_record.get("document_id") or document_id,
        "text_unit_id": route_record.get("text_unit_id"),
        "group_id": route_record.get("group_id"),
        "source_group_id": route_record.get("source_group_id"),
        "source_layer_group_id": route_record.get("source_layer_group_id"),
        "layer": route_record.get("layer"),
        "visual_class": get_visual_class(route_record),
        "recommended_route": classification.get("recommended_route"),
        "n03_node": classification.get("node"),
        "n03_model": classification.get("model"),
        "n03_model_version": classification.get("model_version"),
        "classification_crop_path": route_record.get("classification_crop_path"),
        "classification_crop_source": route_record.get(
            "classification_crop_source"
        ),
        "classification_crop_policy": route_record.get(
            "classification_crop_policy"
        ),
        "analysis_crop_path": route_record.get("analysis_crop_path"),
        "analysis_mask_crop_path": route_record.get("analysis_mask_crop_path"),
        "context_crop_path": route_record.get("context_crop_path"),
        "original_crop_path": route_record.get("original_crop_path"),
        "refined_crop_path": route_record.get("refined_crop_path"),
        "routed_crop_path": route_record.get("routed_crop_path"),
        "source_crop_path": route_record.get("source_crop_path"),
        "n05_selected_crop_path": selected_crop_path,
        "n05_selected_crop_source": selected_crop_source,
        "n05_copied_crop_path": copied_crop_path,
        "scribetrace_candidate": True,
        "scribetrace_mask_crop_path": route_record.get("analysis_mask_crop_path"),
        "scribetrace_visual_crop_path": copied_crop_path or selected_crop_path,
        "scribetrace_context_crop_path": route_record.get("context_crop_path"),
        "scribetrace_output_dir": folders["scribetrace"],
        "mask_source": route_record.get("mask_source"),
        "visual_layer_source": route_record.get("visual_layer_source"),
        "document_bbox": build_document_bbox(route_record),
        "crop_bbox": build_crop_bbox(route_record),
        "final_bbox": _normalized_bbox(route_record.get("final_bbox")),
        "n03_scores": classification.get("scores", {}),
        "n03_thresholds": classification.get("thresholds", {}),
        "layer_hypothesis": route_record.get("layer_hypothesis"),
        "role_guess": route_record.get("role_guess"),
        "minos_mode": route_record.get("minos_mode"),
        "is_final_text_candidate": route_record.get(
            "is_final_text_candidate",
            True,
        ),
        "preserve_as_evidence": route_record.get("preserve_as_evidence", False),
        "force_handwritten_ocr": route_record.get(
            "force_handwritten_ocr",
            False,
        ),
        "correction_role": route_record.get("correction_role"),
        "replaces_blue_source_group_ids": route_record.get(
            "replaces_blue_source_group_ids",
            [],
        ),
        "correction_evidence": route_record.get("correction_evidence", []),
        "handwriting_profile": build_default_handwriting_profile(),
        "engine_routing": build_default_engine_routing(),
        "edge_cases": build_default_edge_cases(),
        "handwritten_ocr": build_placeholder_handwriting_result(),
    }


def build_scribetrace_context_from_unit(handwritten_text_unit):
    """Build the existing whole-unit context expected by ScribeTrace."""
    return {
        "document_id": handwritten_text_unit.get("document_id"),
        "text_unit_id": handwritten_text_unit.get("text_unit_id"),
        "source_group_id": handwritten_text_unit.get("source_group_id"),
        "source_layer_group_id": handwritten_text_unit.get(
            "source_layer_group_id"
        ),
        "layer": handwritten_text_unit.get("layer"),
        "document_bbox": handwritten_text_unit.get("document_bbox"),
        "final_bbox": handwritten_text_unit.get("final_bbox"),
        "scribetrace_mask_crop_path": handwritten_text_unit.get(
            "scribetrace_mask_crop_path"
        ),
        "scribetrace_visual_crop_path": handwritten_text_unit.get(
            "scribetrace_visual_crop_path"
        ),
        "scribetrace_context_crop_path": handwritten_text_unit.get(
            "scribetrace_context_crop_path"
        ),
        "scribetrace_output_dir": handwritten_text_unit.get(
            "scribetrace_output_dir"
        ),
    }


def run_scribetrace_for_text_unit(handwritten_text_unit, settings):
    """Run existing ScribeTrace on the original whole-unit crop only."""
    crop_path = (
        handwritten_text_unit.get("n05_copied_crop_path")
        or handwritten_text_unit.get("n05_selected_crop_path")
    )
    if not crop_path:
        return {
            "expert_name": "scribetrace",
            "attempted": False,
            "status": "skipped",
            "crop_path": None,
            "candidates": [],
            "evidence": {},
            "error": "No crop path available for ScribeTrace.",
        }

    scribetrace_settings = settings.get("experts", {}).get("scribetrace", {})
    return recognize_scribetrace(
        crop_path=crop_path,
        context=build_scribetrace_context_from_unit(handwritten_text_unit),
        settings=scribetrace_settings,
    )


def build_word_level_context_from_unit(handwritten_text_unit):
    """Build context for whole-word OCR evidence."""

    return {
        "document_id": handwritten_text_unit.get("document_id"),
        "text_unit_id": handwritten_text_unit.get("text_unit_id"),
        "group_id": handwritten_text_unit.get("group_id"),
        "layer": handwritten_text_unit.get("layer"),
        "visual_class": handwritten_text_unit.get("visual_class"),
        "document_bbox": handwritten_text_unit.get("document_bbox"),
        "crop_bbox": handwritten_text_unit.get("crop_bbox"),
        "character_unit_proposal_summary": {
            "status": handwritten_text_unit.get("character_unit_proposal", {}).get(
                "status"
            ),
            "recovery_needed": handwritten_text_unit.get(
                "character_unit_proposal", {}
            ).get("recovery_needed"),
            "hypothesis_count": len(
                handwritten_text_unit.get("character_unit_proposal", {}).get(
                    "segmentation_hypotheses",
                    [],
                )
            ),
        },
    }


def run_word_level_ocr_for_text_unit(handwritten_text_unit, settings):
    """Run word-level OCR on the original whole text-unit crop."""

    crop_path = (
        handwritten_text_unit.get("n05_copied_crop_path")
        or handwritten_text_unit.get("n05_selected_crop_path")
    )
    if not crop_path:
        return {
            "expert_name": "word_level_ocr",
            "attempted": False,
            "status": "skipped",
            "crop_path": None,
            "context": build_word_level_context_from_unit(handwritten_text_unit),
            "candidates": [],
            "evidence": None,
            "error": "No crop path available for word-level OCR.",
        }

    word_settings = settings.get("experts", {}).get("word_level_ocr", {})
    return recognize_word_level(
        crop_path=crop_path,
        context=build_word_level_context_from_unit(handwritten_text_unit),
        settings=word_settings,
    )


def attach_word_level_result(handwritten_text_unit, word_level_result):
    """Attach whole-word OCR evidence for assembly consumption."""

    handwritten_text_unit.setdefault("expert_outputs", {})
    handwritten_text_unit["expert_outputs"]["word_level_ocr"] = word_level_result
    evidence = word_level_result.get("evidence") or {}
    prediction = evidence.get("prediction") or {}
    handwritten_text_unit["word_level_ocr"] = {
        "attempted": bool(word_level_result.get("attempted")),
        "status": word_level_result.get("status"),
        "text": prediction.get("text"),
        "confidence": prediction.get("confidence"),
        "decoded_length": prediction.get("decoded_length"),
        "predicted_length": prediction.get("predicted_length"),
        "predicted_bridge_count": prediction.get("predicted_bridge_count"),
        "split_line_candidate_count": len(
            prediction.get("split_line_candidates") or []
        ),
        "token_count": len(prediction.get("tokens") or []),
        "error": word_level_result.get("error"),
    }
    return handwritten_text_unit


def attach_scribetrace_result(handwritten_text_unit, scribetrace_result):
    """Attach whole-unit ScribeTrace evidence without changing OCR candidates."""
    handwritten_text_unit.setdefault("expert_outputs", {})
    handwritten_text_unit["expert_outputs"]["scribetrace"] = scribetrace_result
    evidence = scribetrace_result.get("evidence", {})
    handwritten_text_unit["scribetrace_rf"] = {
        "attempted": bool(scribetrace_result.get("attempted")),
        "status": scribetrace_result.get("status"),
        "letter_level_model": "scribetrace_random_forest_v0_2_1",
        "unit_level_warning": (
            "N05 input is a word/text-unit crop. Segment hypotheses are not "
            "wired to experts yet."
        ),
        "top5_letter_candidates_for_unit": evidence.get(
            "rf_letter_candidates_for_unit",
            [],
        ),
        "error": scribetrace_result.get("error") or evidence.get("rf_error"),
    }
    return handwritten_text_unit


def summarize_handwritten_text_map(
    visual_routes_payload,
    selected_routes,
    handwritten_text_units,
    skipped_records,
    failed_records,
):
    """Build compact N05 route, placeholder, and recovery statistics."""
    return {
        "total_visual_routes_read": len(visual_routes_payload.get("routes", [])),
        "handwriting_candidates_selected": len(selected_routes),
        "handwriting_only_count": sum(
            get_visual_class(route) == "handwriting_only"
            for route in selected_routes
        ),
        "mixed_count": sum(
            get_visual_class(route) == "mixed" for route in selected_routes
        ),
        "fallback_from_printed_only_count": 0,
        "handwritten_text_units_count": len(handwritten_text_units),
        "skipped_count": len(skipped_records),
        "failed_count": len(failed_records),
        "ocr_attempted_count": sum(
            bool(unit.get("handwritten_ocr", {}).get("attempted"))
            for unit in handwritten_text_units
        ),
        "raw_candidate_count": sum(
            len(unit.get("handwritten_ocr", {}).get("candidates", []))
            for unit in handwritten_text_units
        ),
        "ocr_failed_count": sum(
            unit.get("handwritten_ocr", {}).get("status") == "failed"
            for unit in handwritten_text_units
        ),
        "placeholder_count": sum(
            not unit.get("handwritten_ocr", {}).get("attempted", False)
            for unit in handwritten_text_units
        ),
        "character_unit_recovery_count": sum(
            bool(unit.get("character_unit_proposal", {}).get("recovery_needed"))
            for unit in handwritten_text_units
        ),
        "character_unit_split_hypothesis_count": sum(
            max(
                0,
                len(
                    unit.get("character_unit_proposal", {}).get(
                        "segmentation_hypotheses",
                        [],
                    )
                )
                - 1,
            )
            for unit in handwritten_text_units
        ),
        "character_unit_split_hint_count": sum(
            len(
                unit.get("character_unit_proposal", {}).get(
                    "split_hints",
                    [],
                )
            )
            for unit in handwritten_text_units
        ),
        "word_level_ocr_attempted_count": sum(
            bool(unit.get("word_level_ocr", {}).get("attempted"))
            for unit in handwritten_text_units
        ),
        "word_level_ocr_completed_count": sum(
            unit.get("word_level_ocr", {}).get("status") == "completed"
            for unit in handwritten_text_units
        ),
        "word_level_ocr_token_count": sum(
            int(unit.get("word_level_ocr", {}).get("token_count") or 0)
            for unit in handwritten_text_units
        ),
    }


def print_summary(document_id, summary, metadata_path, assembly_path=None):
    """Print a short N05 completion summary."""
    print("-------------------------")
    print("N05 handwritten text map completed.")
    print("Document:", document_id)
    print("Visual routes read:", summary["total_visual_routes_read"])
    print("Handwriting candidates:", summary["handwriting_candidates_selected"])
    print("Handwritten text units:", summary["handwritten_text_units_count"])
    print("Recovery flagged:", summary["character_unit_recovery_count"])
    print("Split hypotheses:", summary["character_unit_split_hypothesis_count"])
    print("Trace-validated split hints:", summary["character_unit_split_hint_count"])
    print("Word OCR attempted:", summary["word_level_ocr_attempted_count"])
    print("Word OCR completed:", summary["word_level_ocr_completed_count"])
    print("Skipped:", summary["skipped_count"])
    print("Failed:", summary["failed_count"])
    print("Metadata:", metadata_path)
    if assembly_path:
        print("Assembly:", assembly_path)
    print("-------------------------")


def build_handwriting_expert_map(
    visual_routes_path,
    output_dir,
    settings_path=None,
):
    """Build the N05 map and attach character-unit proposals to every unit."""
    visual_routes_payload = load_visual_routes(visual_routes_path)
    settings = load_settings(settings_path)

    if settings.get("reset_output", True):
        reset_output_dir(output_dir)
    folders = create_output_folders(output_dir)

    document_id = visual_routes_payload.get("document_id", "unknown_document")
    route_records = visual_routes_payload.get("routes", [])
    selected_routes = select_handwriting_candidates(route_records)
    handwritten_text_units = []
    skipped_records = []
    failed_records = []

    for route_record in selected_routes:
        try:
            copied_crop_path, selected_crop_path, selected_crop_source = (
                copy_candidate_crop(
                    route_record,
                    folders,
                    save_copy=settings.get("copy_selected_crops", False),
                )
            )
            if selected_crop_path is None:
                skipped_records.append(
                    {
                        "document_id": document_id,
                        "text_unit_id": route_record.get("text_unit_id"),
                        "group_id": route_record.get("group_id"),
                        "reason": "no_usable_crop_path",
                    }
                )
                continue

            handwritten_text_unit = build_handwritten_text_unit(
                route_record=route_record,
                copied_crop_path=copied_crop_path,
                selected_crop_path=selected_crop_path,
                selected_crop_source=selected_crop_source,
                folders=folders,
                document_id=document_id,
            )

            # Universal pre-expert proposal. Experts continue to receive the
            # original whole-unit crop until hypothesis routing is implemented.
            handwritten_text_unit["character_unit_proposal"] = (
                propose_character_units(
                    handwritten_text_unit=handwritten_text_unit,
                    folders=folders,
                    settings=settings.get("character_unit_proposer", {}),
                )
            )

            word_level_result = run_word_level_ocr_for_text_unit(
                handwritten_text_unit=handwritten_text_unit,
                settings=settings,
            )
            attach_word_level_result(
                handwritten_text_unit=handwritten_text_unit,
                word_level_result=word_level_result,
            )

            scribetrace_result = run_scribetrace_for_text_unit(
                handwritten_text_unit=handwritten_text_unit,
                settings=settings,
            )
            attach_scribetrace_result(
                handwritten_text_unit=handwritten_text_unit,
                scribetrace_result=scribetrace_result,
            )
            handwritten_text_units.append(handwritten_text_unit)
        except Exception as error:
            failed_records.append(
                {
                    "document_id": document_id,
                    "text_unit_id": route_record.get("text_unit_id"),
                    "group_id": route_record.get("group_id"),
                    "error": str(error),
                    "route_record": route_record,
                }
            )

    summary = summarize_handwritten_text_map(
        visual_routes_payload=visual_routes_payload,
        selected_routes=selected_routes,
        handwritten_text_units=handwritten_text_units,
        skipped_records=skipped_records,
        failed_records=failed_records,
    )
    assembly_map = build_assembly_map(
        document_id=document_id,
        handwritten_text_units=handwritten_text_units,
        settings=settings.get("assembly", {}),
        output_dir=folders["assembly"],
    )
    assembly_path = os.path.join(
        folders["assembly"],
        f"{document_id}_assembly.json",
    )
    save_json(assembly_map, assembly_path)
    result = {
        "node": NODE_NAME,
        "node_version": NODE_VERSION,
        "orchestrator": ORCHESTRATOR_NAME,
        "document_id": document_id,
        "source_visual_routes_path": os.path.abspath(visual_routes_path),
        "output_dir": os.path.abspath(output_dir),
        "crops_dir": folders["crops"],
        "metadata_dir": folders["metadata"],
        "coordinate_space": "original_document_image",
        "expert_registry": build_expert_registry(settings),
        "summary": summary,
        "assembly": assembly_map,
        "assembly_path": assembly_path,
        "handwritten_text_units": handwritten_text_units,
        "skipped": skipped_records,
        "failed": failed_records,
    }
    metadata_path = os.path.join(
        folders["metadata"],
        f"{document_id}_handwritten_text_map.json",
    )
    save_json(result, metadata_path)
    result["metadata_path"] = metadata_path
    print_summary(document_id, summary, metadata_path, assembly_path=assembly_path)
    return result


__all__ = [
    "attach_scribetrace_result",
    "build_handwriting_expert_map",
    "build_handwritten_text_unit",
    "build_scribetrace_context_from_unit",
    "create_output_folders",
    "run_scribetrace_for_text_unit",
]

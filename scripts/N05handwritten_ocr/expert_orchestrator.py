"""Build the coordinate-aware N05 handwriting mixture-of-experts map."""

import json
import os
import shutil

try:
    from .assembly import build_assembly_map
    from .assembly.combined_output import build_combined_expert_output
    from .assembly.decision_matrix import build_decision_matrix
    from .assembly.evidence_fusion import build_assembly_summary
    from .assembly.letter_matrix import build_letter_matrix
    from .assembly.matrix_envelope import build_matrix_envelope
    from .character_detector import get_expert_manifest as get_character_manifest
    from .character_detector import recognize as recognize_character_detector
    from .scribetrace import get_expert_manifest as get_scribetrace_manifest
    from .scribetrace import recognize as recognize_scribetrace
    from .scribetrain_word_segmenter import propose_scribetrain_word_segments
    from .scrilog import run_scrilog_on_payload
    from .scribejudge import build_scribejudge_overlay
    from .tesseract_ocr import get_expert_manifest as get_tesseract_manifest
    from .word_level_ocr import get_expert_manifest as get_word_level_manifest
    from .word_level_ocr import recognize as recognize_word_level
except ImportError:
    from assembly import build_assembly_map
    from assembly.combined_output import build_combined_expert_output
    from assembly.decision_matrix import build_decision_matrix
    from assembly.evidence_fusion import build_assembly_summary
    from assembly.letter_matrix import build_letter_matrix
    from assembly.matrix_envelope import build_matrix_envelope
    from character_detector import get_expert_manifest as get_character_manifest
    from character_detector import recognize as recognize_character_detector
    from scribetrace import get_expert_manifest as get_scribetrace_manifest
    from scribetrace import recognize as recognize_scribetrace
    from scribetrain_word_segmenter import propose_scribetrain_word_segments
    from scrilog import run_scrilog_on_payload
    from scribejudge import build_scribejudge_overlay
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
    """Build the expert registry without eagerly loading model weights."""
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
        "scrilog_scrististics": {
            "expert_name": "scrilog_scrististics",
            "display_name": "ScriLog + ScriStatistics",
            "enabled": bool(
                expert_settings.get("scrilog_scrististics", {}).get("enabled", False)
            ),
            "implemented": True,
            "status": "symbolic_statistical_geometry_ready",
            "unit_level": "selected_segment",
            "returns_text": False,
        },
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
            "legacy",
            "character_unit_proposer",
        ),
        "character_unit_segments": os.path.join(
            output_dir,
            "legacy",
            "character_unit_proposer",
            "segments",
        ),
        "character_unit_debug": os.path.join(
            output_dir,
            "legacy",
            "character_unit_proposer",
            "debug",
        ),
        "scribetrain_word_segmenter": os.path.join(
            output_dir,
            "scribetrain_word_segmenter",
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


def get_route_layer(route_record):
    """Return the semantic ink layer for one upstream route when available."""

    layer = (
        route_record.get("layer")
        or route_record.get("classification_layer")
        or route_record.get("mask_layer")
    )
    if layer:
        return str(layer)
    visual_source = str(route_record.get("visual_layer_source") or "")
    if visual_source.endswith("_ink_layer"):
        return visual_source[: -len("_ink_layer")]
    mask_source = str(route_record.get("mask_source") or "")
    if mask_source.endswith("_continuity_mask"):
        return mask_source[: -len("_continuity_mask")]
    if mask_source.endswith("_ink_mask"):
        return mask_source[: -len("_ink_mask")]
    return None


def route_layer_allowed(route_record, settings):
    """Return whether this route's layer is currently allowed into N05."""

    allowed_layers = settings.get("allowed_layers")
    if not allowed_layers:
        return True
    normalized = {str(layer).lower() for layer in allowed_layers}
    route_layer = get_route_layer(route_record)
    return bool(route_layer and route_layer.lower() in normalized)


def select_handwriting_candidates(route_records, settings=None):
    """Select N03 handwriting-only and mixed records deterministically."""
    settings = settings or {}
    return [
        route_record
        for route_record in route_records
        if should_send_to_handwritten_ocr(route_record)
        and route_layer_allowed(route_record, settings)
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


def build_character_detector_context(document_id, entry, path, segment, position):
    """Build context for one segment-level CNN inference call."""
    return {
        "document_id": document_id,
        "text_unit_id": entry.get("text_unit_id"),
        "group_id": entry.get("group_id"),
        "path_id": path.get("path_id"),
        "path_type": path.get("path_type") or path.get("type"),
        "segment_id": segment.get("segment_id"),
        "segment_position": position,
        "segment_bbox": segment.get("bbox"),
        "source": "n05_assembly_selected_segment",
    }


def build_segment_scribetrace_context(document_id, entry, path, segment, position):
    """Build context for one selected segment ScribeTrace pass."""

    crop_path = segment.get("mask_crop_path") or segment.get("visual_crop_path")
    assembly_dir = None
    if crop_path:
        # Expected shape: assembly/segments/mask/<file>.png
        assembly_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(crop_path)))
        )
    return {
        "document_id": document_id,
        "text_unit_id": (
            f"{entry.get('text_unit_id')}_"
            f"{path.get('path_id')}_"
            f"{segment.get('segment_id')}"
        ),
        "group_id": entry.get("group_id"),
        "source_group_id": entry.get("text_unit_id"),
        "scribetrace_mask_crop_path": segment.get("mask_crop_path"),
        "scribetrace_visual_crop_path": segment.get("visual_crop_path"),
        "scribetrace_output_dir": (
            os.path.join(assembly_dir, "scribetrace_segments")
            if assembly_dir
            else None
        ),
        "condition_verdict": (
            (segment.get("condition") or {}).get("verdict")
            if isinstance(segment.get("condition"), dict)
            else None
        ),
        "routing_advice": (
            (segment.get("condition") or {}).get("routing")
            if isinstance(segment.get("condition"), dict)
            else None
        ),
        "assembly_path_id": path.get("path_id"),
        "path_type": path.get("path_type") or path.get("type"),
        "segment_id": segment.get("segment_id"),
        "segment_position": position,
        "segment_bbox": segment.get("bbox"),
        "source": "n05_assembly_selected_segment",
    }


def _selected_segment_records(assembly_map):
    """Yield selected assembly segments with their parent records."""

    for entry in assembly_map.get("segmentation_matrix") or []:
        for path in entry.get("paths") or []:
            if path.get("status") != "selected_for_v0_2_segment_artifacts":
                continue
            for position, segment in enumerate(path.get("segments") or []):
                yield entry, path, segment, position


def _find_segment_by_job(assembly_map, job):
    """Find a segment matching an expert job row."""

    for entry, path, segment, _position in _selected_segment_records(assembly_map):
        if entry.get("text_unit_id") != job.get("text_unit_id"):
            continue
        if path.get("path_id") != job.get("path_id"):
            continue
        if segment.get("segment_id") == job.get("segment_id"):
            return segment
    return None


def _mark_job_executed(assembly_map, expert_name, result_getter):
    """Mirror segment expert outputs into the queued expert records."""

    for job in assembly_map.get("expert_job_queue") or []:
        if expert_name not in (job.get("experts_needed") or []):
            continue
        segment = _find_segment_by_job(assembly_map, job)
        if not segment:
            continue
        result = result_getter(segment)
        if not result:
            continue
        job.setdefault("executed_experts", {})
        job["executed_experts"][expert_name] = result


def run_scribetrace_scrilog_for_assembly_segments(assembly_map, settings):
    """Run segment-level ScribeTrace, then ScriLog/ScriStatistics on its payload."""

    expert_settings = settings.get("experts", {})
    scribetrace_settings = expert_settings.get("scribetrace", {})
    scrilog_settings = expert_settings.get("scrilog_scrististics", {})
    scribetrace_enabled = bool(scribetrace_settings.get("enabled", False))
    scrilog_enabled = bool(scrilog_settings.get("enabled", False))
    document_id = assembly_map.get("document_id")
    attempted = 0
    scribetrace_completed = 0
    scrilog_completed = 0
    failed = 0

    if not scribetrace_enabled and not scrilog_enabled:
        return {
            "enabled": False,
            "attempted_segments": 0,
            "scribetrace_completed_segments": 0,
            "scrilog_completed_segments": 0,
            "failed_segments": 0,
        }

    for entry, path, segment, position in _selected_segment_records(assembly_map):
        crop_path = segment.get("mask_crop_path") or segment.get("visual_crop_path")
        if not crop_path:
            continue
        attempted += 1
        segment.setdefault("expert_outputs", {})

        scribetrace_result = None
        if scribetrace_enabled:
            scribetrace_result = recognize_scribetrace(
                crop_path=crop_path,
                context=build_segment_scribetrace_context(
                    document_id=document_id,
                    entry=entry,
                    path=path,
                    segment=segment,
                    position=position,
                ),
                settings=scribetrace_settings,
            )
            segment["expert_outputs"]["scribetrace"] = scribetrace_result
            segment["scribetrace"] = {
                "attempted": bool(scribetrace_result.get("attempted")),
                "status": scribetrace_result.get("status"),
                "candidate_count": len(
                    (
                        scribetrace_result.get("evidence", {})
                        .get("rf_letter_candidates_for_unit", [])
                    )
                ),
                "top_candidate": (
                    (
                        scribetrace_result.get("evidence", {})
                        .get("rf_letter_candidates_for_unit", [{}])
                    )[0].get("char")
                    if (
                        scribetrace_result.get("evidence", {})
                        .get("rf_letter_candidates_for_unit")
                    )
                    else None
                ),
                "error": scribetrace_result.get("error"),
            }
            if scribetrace_result.get("status") in {"completed", "completed_limited"}:
                scribetrace_completed += 1
            elif scribetrace_result.get("status") != "disabled":
                failed += 1

        if scrilog_enabled:
            scribetrace_payload = (
                scribetrace_result.get("evidence")
                if isinstance(scribetrace_result, dict)
                else None
            )
            if not scribetrace_payload:
                segment["expert_outputs"]["scrilog_scrististics"] = {
                    "expert_name": "scrilog_scrististics",
                    "attempted": False,
                    "status": "skipped",
                    "evidence": {},
                    "candidate_effects": [],
                    "error": "ScribeTrace payload missing for ScriLog.",
                }
                continue
            try:
                scrilog_result = run_scrilog_on_payload(scribetrace_payload)
                wrapped_result = {
                    "expert_name": "scrilog_scrististics",
                    "attempted": True,
                    "status": "completed",
                    "evidence": scrilog_result,
                    "candidate_effects": scrilog_result.get("candidate_effects", []),
                    "boosted_candidates": scrilog_result.get("boosted_candidates", []),
                    "blocked_candidates": scrilog_result.get("blocked_candidates", []),
                    "weakened_candidates": scrilog_result.get("weakened_candidates", []),
                    "error": None,
                }
                scrilog_completed += 1
            except Exception as error:
                wrapped_result = {
                    "expert_name": "scrilog_scrististics",
                    "attempted": True,
                    "status": "failed",
                    "evidence": {},
                    "candidate_effects": [],
                    "error": str(error),
                }
                failed += 1
            segment["expert_outputs"]["scrilog_scrististics"] = wrapped_result
            segment["scrilog_scrististics"] = {
                "attempted": bool(wrapped_result.get("attempted")),
                "status": wrapped_result.get("status"),
                "candidate_effect_count": len(wrapped_result.get("candidate_effects") or []),
                "boosted_candidate_count": len(wrapped_result.get("boosted_candidates") or []),
                "error": wrapped_result.get("error"),
            }

    _mark_job_executed(
        assembly_map,
        "scribetrace",
        lambda segment: {
            "status": segment.get("expert_outputs", {}).get("scribetrace", {}).get("status"),
            "candidate_count": len(
                segment.get("expert_outputs", {})
                .get("scribetrace", {})
                .get("evidence", {})
                .get("rf_letter_candidates_for_unit", [])
            ),
            "top_candidate": (
                (
                    segment.get("expert_outputs", {})
                    .get("scribetrace", {})
                    .get("evidence", {})
                    .get("rf_letter_candidates_for_unit", [{}])
                )[0].get("char")
                if segment.get("expert_outputs", {})
                .get("scribetrace", {})
                .get("evidence", {})
                .get("rf_letter_candidates_for_unit")
                else None
            ),
        },
    )
    _mark_job_executed(
        assembly_map,
        "scrilog",
        lambda segment: {
            "status": segment.get("expert_outputs", {})
            .get("scrilog_scrististics", {})
            .get("status"),
            "candidate_effect_count": len(
                segment.get("expert_outputs", {})
                .get("scrilog_scrististics", {})
                .get("candidate_effects", [])
            ),
        },
    )
    _mark_job_executed(
        assembly_map,
        "scrististics",
        lambda segment: {
            "status": segment.get("expert_outputs", {})
            .get("scrilog_scrististics", {})
            .get("status"),
            "statistical_evidence_available": bool(
                segment.get("expert_outputs", {})
                .get("scrilog_scrististics", {})
                .get("evidence", {})
                .get("statistical_evidence")
            ),
        },
    )

    summary = {
        "enabled": True,
        "attempted_segments": attempted,
        "scribetrace_completed_segments": scribetrace_completed,
        "scrilog_completed_segments": scrilog_completed,
        "failed_segments": failed,
    }
    assembly_map.setdefault("summary", {})
    assembly_map["summary"]["scribetrace_scrilog_segments"] = summary
    return summary


def run_character_detector_for_assembly_segments(assembly_map, settings):
    """Run the pixel CNN on selected N05 assembly segment masks.

    The character detector is intentionally segment-level. Running it on a
    whole word crop makes the CNN loud and misleading, while selected segment
    masks match the glyph-level training contract.
    """

    character_settings = settings.get("experts", {}).get("character_detector", {})
    if not bool(character_settings.get("enabled", False)):
        return {
            "enabled": False,
            "attempted_segments": 0,
            "completed_segments": 0,
            "failed_segments": 0,
        }

    document_id = assembly_map.get("document_id")
    attempted = 0
    completed = 0
    failed = 0

    for entry in assembly_map.get("segmentation_matrix") or []:
        for path in entry.get("paths") or []:
            if path.get("status") != "selected_for_v0_2_segment_artifacts":
                continue
            for position, segment in enumerate(path.get("segments") or []):
                crop_path = segment.get("mask_crop_path") or segment.get("visual_crop_path")
                if not crop_path:
                    continue
                attempted += 1
                result = recognize_character_detector(
                    crop_path=crop_path,
                    context=build_character_detector_context(
                        document_id=document_id,
                        entry=entry,
                        path=path,
                        segment=segment,
                        position=position,
                    ),
                    settings=character_settings,
                )
                segment.setdefault("expert_outputs", {})
                segment["expert_outputs"]["character_detector"] = result
                segment["character_detector"] = {
                    "attempted": bool(result.get("attempted")),
                    "status": result.get("status"),
                    "top_candidate": (
                        (result.get("candidates") or [{}])[0].get("label")
                        if result.get("candidates")
                        else None
                    ),
                    "top_confidence": (
                        (result.get("candidates") or [{}])[0].get("confidence")
                        if result.get("candidates")
                        else None
                    ),
                    "candidate_count": len(result.get("candidates") or []),
                    "error": result.get("error"),
                }
                if result.get("status") == "completed":
                    completed += 1
                else:
                    failed += 1

    for job in assembly_map.get("expert_job_queue") or []:
        if "character_detector" not in (job.get("experts_needed") or []):
            continue
        matching_result = None
        for entry in assembly_map.get("segmentation_matrix") or []:
            if entry.get("text_unit_id") != job.get("text_unit_id"):
                continue
            for path in entry.get("paths") or []:
                if path.get("path_id") != job.get("path_id"):
                    continue
                for segment in path.get("segments") or []:
                    if segment.get("segment_id") == job.get("segment_id"):
                        matching_result = (
                            segment.get("expert_outputs", {})
                            .get("character_detector")
                        )
                        break
        if matching_result:
            job.setdefault("executed_experts", {})
            job["executed_experts"]["character_detector"] = {
                "status": matching_result.get("status"),
                "candidate_count": len(matching_result.get("candidates") or []),
                "top_candidate": (
                    (matching_result.get("candidates") or [{}])[0].get("label")
                    if matching_result.get("candidates")
                    else None
                ),
            }

    summary = {
        "enabled": True,
        "attempted_segments": attempted,
        "completed_segments": completed,
        "failed_segments": failed,
    }
    assembly_map.setdefault("summary", {})
    assembly_map["summary"]["character_detector_segments"] = summary
    return summary



def _job_key(job_or_segment_record):
    """Build a stable key for matching queued expert jobs after matrix rebuilds."""

    if not isinstance(job_or_segment_record, dict):
        return None
    return (
        str(job_or_segment_record.get("text_unit_id")),
        str(job_or_segment_record.get("path_id")),
        str(job_or_segment_record.get("segment_id")),
    )


def _index_executed_job_records(expert_job_queue):
    """Index existing executed_experts records before rebuilding the envelope."""

    index = {}
    for job in expert_job_queue or []:
        key = _job_key(job)
        if key is None:
            continue
        executed = job.get("executed_experts")
        if executed:
            index[key] = executed
    return index


def _restore_executed_job_records(expert_job_queue, executed_index):
    """Restore executed_experts onto newly rebuilt job records."""

    restored = 0
    for job in expert_job_queue or []:
        key = _job_key(job)
        if key in executed_index:
            job["executed_experts"] = executed_index[key]
            restored += 1
    return restored


def rebuild_assembly_matrices_after_segment_experts(
    assembly_map,
    handwritten_text_units,
    settings,
):
    """Refresh letter/matrix surfaces after selected-segment experts run.

    This is the bridge between:
        segment["expert_outputs"]  ->  letter_matrix / matrices / summary

    Without this step, segment-level CNN/ScribeTrace evidence exists in the JSON
    but does not affect the letter matrix.
    """

    assembly_settings = settings.get("assembly", {})
    letter_settings = assembly_settings.get("letter_matrix", {})
    combined_settings = assembly_settings.get("combined_output", {})
    decision_settings = assembly_settings.get("decision_matrix", {})
    scribejudge_settings = assembly_settings.get(
        "scribejudge",
        settings.get("scribejudge", {}),
    )
    segmentation_matrix = assembly_map.get("segmentation_matrix") or []

    previous_executed_jobs = _index_executed_job_records(
        assembly_map.get("expert_job_queue") or []
    )
    previous_summary = dict(assembly_map.get("summary") or {})

    letter_matrix = build_letter_matrix(
        handwritten_text_units,
        segmentation_matrix,
        settings=letter_settings,
    )

    envelope = build_matrix_envelope(
        document_id=assembly_map.get("document_id"),
        unit_count=len(handwritten_text_units),
        segmentation_matrix=segmentation_matrix,
        letter_matrix=letter_matrix,
    )

    restored_job_count = _restore_executed_job_records(
        envelope.get("expert_job_queue") or [],
        previous_executed_jobs,
    )

    assembly_map["letter_matrix"] = letter_matrix
    assembly_map.update(envelope)
    assembly_map["combined_expert_output"] = build_combined_expert_output(
        assembly_map=assembly_map,
        settings=combined_settings,
    )
    assembly_map.setdefault("matrices", {})["combined_output"] = {
        "version": assembly_map["combined_expert_output"].get("version"),
        "status": assembly_map["combined_expert_output"].get("status"),
        "inputs": ["letter_evidence"],
        "outputs": ["n06_word_tokens", "position_candidate_backups"],
        "rows": assembly_map["combined_expert_output"].get("word_tokens", []),
    }
    assembly_map["decision_matrix"] = build_decision_matrix(
        combined_expert_output=assembly_map["combined_expert_output"],
        settings=decision_settings,
    )
    assembly_map.setdefault("matrices", {})["decision"] = {
        "version": assembly_map["decision_matrix"].get("version"),
        "status": assembly_map["decision_matrix"].get("status"),
        "inputs": ["combined_output"],
        "outputs": ["provisional_word_candidates"],
        "rows": assembly_map["decision_matrix"].get("rows", []),
    }
    project_root = os.path.dirname(os.path.dirname(MODULE_DIR))
    assembly_map["scribejudge_overlay"] = build_scribejudge_overlay(
        assembly_map=assembly_map,
        settings=scribejudge_settings,
        base_dir=project_root,
    )
    assembly_map.setdefault("matrices", {})["scribejudge"] = {
        "version": assembly_map["scribejudge_overlay"].get("version"),
        "status": assembly_map["scribejudge_overlay"].get("status"),
        "inputs": ["decision_matrix", "combined_output", "confusion_history"],
        "outputs": ["advisory_risk_overlay", "future_meta_model_features"],
        "rows": assembly_map["scribejudge_overlay"].get("rows", []),
    }

    rebuilt_summary = build_assembly_summary(
        assembly_map.get("segmentation_matrix") or [],
        letter_matrix,
        matrices=assembly_map.get("matrices", {}),
        expert_job_queue=assembly_map.get("expert_job_queue", []),
    )
    rebuilt_summary["combined_output"] = assembly_map[
        "combined_expert_output"
    ].get("summary", {})
    rebuilt_summary["decision_matrix"] = assembly_map[
        "decision_matrix"
    ].get("summary", {})
    rebuilt_summary["scribejudge_overlay"] = assembly_map[
        "scribejudge_overlay"
    ].get("summary", {})

    # Preserve runtime summaries already written by segment expert runners.
    for key, value in previous_summary.items():
        if key.endswith("_segments") or key in {
            "character_detector_segments",
            "scribetrace_segments",
            "segment_expert_rebuild",
        }:
            rebuilt_summary[key] = value

    rebuilt_summary["segment_expert_rebuild"] = {
        "status": "completed",
        "letter_matrix_rebuilt": True,
        "matrix_envelope_rebuilt": True,
        "restored_executed_job_count": restored_job_count,
    }

    assembly_map["summary"] = rebuilt_summary
    assembly_map["status"] = "assembled_after_segment_experts"
    return assembly_map

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
    selected_routes = select_handwriting_candidates(
        route_records,
        settings=settings.get("input_filter", {}),
    )
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

            word_level_result = run_word_level_ocr_for_text_unit(
                handwritten_text_unit=handwritten_text_unit,
                settings=settings,
            )
            attach_word_level_result(
                handwritten_text_unit=handwritten_text_unit,
                word_level_result=word_level_result,
            )

            handwritten_text_unit["scribetrain_word_segmentation"] = (
                propose_scribetrain_word_segments(
                    handwritten_text_unit=handwritten_text_unit,
                    settings=settings.get("scribetrain_word_segmenter", {}),
                )
            )
            handwritten_text_unit["character_unit_proposal"] = {
                "schema_version": "n05_character_unit_proposal_legacy_v1",
                "status": "legacy_disabled",
                "active": False,
                "reason": "Replaced by scribetrain_word_segmenter as the primary splitter.",
            }

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
    scribetrace_scrilog_segment_summary = run_scribetrace_scrilog_for_assembly_segments(
        assembly_map=assembly_map,
        settings=settings,
    )
    character_detector_segment_summary = run_character_detector_for_assembly_segments(
        assembly_map=assembly_map,
        settings=settings,
    )
    assembly_map = rebuild_assembly_matrices_after_segment_experts(
        assembly_map=assembly_map,
        handwritten_text_units=handwritten_text_units,
        settings=settings,
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
        "scribetrace_scrilog_segment_summary": scribetrace_scrilog_segment_summary,
        "character_detector_segment_summary": character_detector_segment_summary,
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
    "rebuild_assembly_matrices_after_segment_experts",
    "run_character_detector_for_assembly_segments",
    "run_scribetrace_scrilog_for_assembly_segments",
    "run_scribetrace_for_text_unit",
]

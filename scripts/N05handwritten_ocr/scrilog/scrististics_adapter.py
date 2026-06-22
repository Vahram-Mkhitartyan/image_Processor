"""Translate empirical Scrististics profiles into soft ScriLog evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

from .results import ScriLogCandidateEffect


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "datasets" / "scrististics"
FEATURE_WEIGHTS = {"high": 1.0, "medium": 0.55, "low": 0.2, "unknown": 0.1}
OBSERVATION_FEATURES = {
    "visual_ink_holes": "ink_hole_count",
    "closed_skeleton_loops": "closed_loop_count",
    "endpoints": "endpoint_count",
    "junction_clusters": "junction_cluster_count",
    "trace_paths": "path_count",
    "components": "component_count",
    "isolated_points": "isolated_point_count",
    "short_paths": "short_path_count",
    "endpoint_top_left": ("endpoint_quadrants", "top_left"),
    "endpoint_top_right": ("endpoint_quadrants", "top_right"),
    "endpoint_bottom_left": ("endpoint_quadrants", "bottom_left"),
    "endpoint_bottom_right": ("endpoint_quadrants", "bottom_right"),
    "junction_top_left": ("junction_quadrants", "top_left"),
    "junction_top_right": ("junction_quadrants", "top_right"),
    "junction_bottom_left": ("junction_quadrants", "bottom_left"),
    "junction_bottom_right": ("junction_quadrants", "bottom_right"),
    "touches_left_border": ("border_contacts", "left"),
    "touches_right_border": ("border_contacts", "right"),
    "touches_top_border": ("border_contacts", "top"),
    "touches_bottom_border": ("border_contacts", "bottom"),
    "wide_shape": ("derived_families", "is_wide"),
}

_PROFILE_CACHE: Dict[Path, tuple[int, dict]] = {}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalized_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _newest_profile(profile_dir: Path = DEFAULT_PROFILE_DIR) -> Optional[Path]:
    candidates = sorted(
        profile_dir.glob("empirical_profiles*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_profile(path: Path) -> dict:
    resolved = path.resolve()
    mtime = resolved.stat().st_mtime_ns
    cached = _PROFILE_CACHE.get(resolved)
    if cached and cached[0] == mtime:
        return cached[1]
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _PROFILE_CACHE[resolved] = (mtime, payload)
    return payload


def _selected_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    reconstruction = _safe_dict(payload.get("reconstruction"))
    selected = reconstruction.get("selected_scrilog_observation")
    if isinstance(selected, dict) and selected:
        return selected

    metrics = _safe_dict(payload.get("metrics"))
    observation = metrics.get("scrilog_observation")
    if isinstance(observation, dict) and observation:
        return observation

    direct = payload.get("scrilog_observation")
    return direct if isinstance(direct, dict) else {}


def _extract_observed_features(observation: Dict[str, Any]) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for feature_name, source in OBSERVATION_FEATURES.items():
        if isinstance(source, tuple):
            container = _safe_dict(observation.get(source[0]))
            value = container.get(source[1])
        else:
            value = observation.get(source)
        if value is not None:
            features[feature_name] = value
    return features


def _distribution_probability(distribution: dict, observed_value: Any) -> float:
    """Return smoothed empirical probability for one discrete value."""
    rows = distribution.get("values") or []
    total = max(0, int(distribution.get("total", 0)))
    observed_key = _normalized_value(observed_value)
    count = 0
    for row in rows:
        if _normalized_value(row.get("value")) == observed_key:
            count = int(row.get("count", 0))
            break
    category_count = max(2, len(rows) + (1 if count == 0 else 0))
    smoothing = 0.5
    return (count + smoothing) / (total + smoothing * category_count)


class ScrististicsEvidenceAdapter:
    """Score classes statistically without changing ScriLog structural facts."""

    def __init__(
        self,
        profile_path: Optional[str | Path] = None,
        boost_threshold: float = 1.4,
        weaken_threshold: float = 0.35,
        posterior_temperature: float = 3.0,
    ) -> None:
        self.profile_path = (
            Path(profile_path).resolve()
            if profile_path
            else _newest_profile()
        )
        self.boost_threshold = float(boost_threshold)
        self.weaken_threshold = float(weaken_threshold)
        self.posterior_temperature = max(0.1, float(posterior_temperature))

    @property
    def available(self) -> bool:
        return bool(self.profile_path and self.profile_path.is_file())

    def evaluate(self, payload: Dict[str, Any]) -> dict:
        """Return compact class likelihoods and bounded soft effects."""
        if not self.available:
            return {
                "status": "unavailable",
                "reason": "empirical_profile_missing",
                "profile_path": str(self.profile_path) if self.profile_path else None,
                "candidate_effects": [],
                "top_candidates": [],
            }

        observation = _selected_observation(payload)
        observed_features = _extract_observed_features(observation)
        if not observed_features:
            return {
                "status": "not_applicable",
                "reason": "selected_scrilog_observation_missing",
                "profile_path": str(self.profile_path),
                "candidate_effects": [],
                "top_candidates": [],
            }

        profile = _load_profile(self.profile_path)
        class_scores = []
        for label, class_profile in (_safe_dict(profile.get("classes"))).items():
            distributions = _safe_dict(class_profile.get("feature_distributions"))
            weighted_log_probability = 0.0
            total_weight = 0.0
            feature_evidence = []
            for feature_name, observed_value in observed_features.items():
                distribution = _safe_dict(distributions.get(feature_name))
                if not distribution:
                    continue
                probability = _distribution_probability(distribution, observed_value)
                importance = str(distribution.get("importance", "unknown"))
                modal_rows = distribution.get("values") or []
                modal_support = (
                    float(modal_rows[0].get("percent", 0)) / 100.0
                    if modal_rows else 0.0
                )
                reliability = max(0.15, modal_support)
                weight = FEATURE_WEIGHTS.get(importance, 0.1) * reliability
                weighted_log_probability += weight * math.log(max(probability, 1e-12))
                total_weight += weight
                feature_evidence.append(
                    {
                        "feature": feature_name,
                        "observed": observed_value,
                        "probability": round(probability, 6),
                        "reliability": round(reliability, 4),
                        "importance": importance,
                    }
                )

            compatibility = (
                math.exp(weighted_log_probability / total_weight)
                if total_weight else 0.0
            )
            class_scores.append(
                {
                    "class_id": str(class_profile.get("raw_class_id", "unknown")),
                    "label": str(class_profile.get("label", label)),
                    "log_score": weighted_log_probability,
                    "compatibility": compatibility,
                    "coverage": len(feature_evidence),
                    "feature_evidence": feature_evidence,
                }
            )

        maximum_log_score = max(
            (row["log_score"] for row in class_scores),
            default=0.0,
        )
        posterior_weights = [
            math.exp(
                (row["log_score"] - maximum_log_score)
                / self.posterior_temperature
            )
            for row in class_scores
        ]
        posterior_total = sum(posterior_weights) or 1.0
        uniform_probability = 1.0 / max(1, len(class_scores))
        for row, posterior_weight in zip(class_scores, posterior_weights):
            row["likelihood"] = posterior_weight / posterior_total
            row["relative_to_uniform"] = (
                row["likelihood"] / uniform_probability
            )

        class_scores.sort(
            key=lambda row: (
                -row["likelihood"],
                int(row["class_id"]) if row["class_id"].isdigit() else math.inf,
                row["label"],
            )
        )
        top_candidates = []
        for rank, row in enumerate(class_scores[:5], start=1):
            top_candidates.append(
                {
                    **row,
                    "rank": rank,
                    "likelihood": round(row["likelihood"], 6),
                    "compatibility": round(row["compatibility"], 6),
                    "relative_to_uniform": round(row["relative_to_uniform"], 4),
                    "feature_evidence": sorted(
                        row["feature_evidence"],
                        key=lambda evidence: (
                            -FEATURE_WEIGHTS.get(evidence["importance"], 0.1),
                            -evidence["reliability"],
                            evidence["feature"],
                        ),
                    )[:6],
                }
            )

        effects = []
        effect_rows = [
            (row, "boost")
            for row in class_scores[:5]
            if row["relative_to_uniform"] >= self.boost_threshold
        ]
        effect_rows.extend(
            (row, "weaken")
            for row in reversed(class_scores[-5:])
            if row["relative_to_uniform"] <= self.weaken_threshold
        )
        for row, effect in effect_rows:
            likelihood = row["likelihood"]
            relative = row["relative_to_uniform"]
            if effect == "boost":
                strength = min(1.0, (relative - self.boost_threshold) / 3.0)
            else:
                strength = min(1.0, (self.weaken_threshold - relative) / self.weaken_threshold)
            effects.append(
                ScriLogCandidateEffect(
                    class_label=row["label"],
                    effect=effect,
                    strength=max(0.01, strength),
                    reason=(
                        f"scrististics_likelihood:{likelihood:.4f};"
                        f"relative_to_uniform:{relative:.3f};"
                        f"coverage:{row['coverage']}"
                    ),
                    provenance="scrististics_empirical_profile",
                )
            )

        # Keep metadata compact while retaining every class score needed by fusion.
        compact_scores = [
            {
                "class_id": row["class_id"],
                "label": row["label"],
                "likelihood": round(row["likelihood"], 6),
                "compatibility": round(row["compatibility"], 6),
                "relative_to_uniform": round(row["relative_to_uniform"], 4),
                "coverage": row["coverage"],
            }
            for row in class_scores
        ]
        return {
            "status": "completed",
            "profile_path": str(self.profile_path),
            "profile_kind": profile.get("profile_kind"),
            "evaluated_class_count": len(class_scores),
            "observed_features": observed_features,
            "policy": {
                "mode": "soft_evidence_only",
                "can_block": False,
                "boost_threshold": self.boost_threshold,
                "weaken_threshold": self.weaken_threshold,
                "posterior_temperature": self.posterior_temperature,
                "importance_weights": dict(FEATURE_WEIGHTS),
            },
            "top_candidates": top_candidates,
            "class_scores": compact_scores,
            "candidate_effects": effects,
        }

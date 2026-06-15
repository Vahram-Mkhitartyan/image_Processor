ScribeTrace Expert
==================

ScribeTrace 4.0: Theoretical Reconstruction
-------------------------------------------
Theoretical reconstruction is an opt-in reasoning layer built on top of normal
ScribeTrace topology. It never overwrites the source mask and never replaces
the normal trace result. The untouched interpretation is always retained as:

    h0_original

The reconstruction cycle is:

    Hypothesize -> Reconstruct -> Retrace -> Verify -> Accept

Current v4 foundation diagnoses:

    disconnected ink components
    abnormal endpoint counts
    isolated skeleton points
    fragmented short paths
    visible holes without matching closed topology
    possible border clipping

Its first bounded repair primitive is endpoint bridging. Candidate endpoints
must be close enough, their outward TracePath tangents must face each other,
and the proposed bridge must mostly cross missing background rather than
existing ink. Every candidate is drawn onto a copied mask and passed through
the complete component, Zhang-Suen, graph, path, landmark, feature, and RF
pipeline again.

A repair cannot be accepted from RF confidence alone. It must exceed the
minimum topology gain, stay below the synthetic-ink budget, and exceed the
combined score:

    topology_weight * topology_score
    + geometry_weight * geometry_score
    + confidence_weight * recognition_score

Topology rewards reduced fragmentation, endpoint repair, isolated-point repair,
and coherent loop recovery. It penalizes invented junctions and new short-path
fragments. Geometry rewards tangent agreement, short bridges, and minimal ink.

The integrated N05 setting remains disabled:

    enable_theoretical_reconstruction: false

Musashi or standalone tests can enable it explicitly. When enabled, the legacy
automatic morphology repair is bypassed so reconstruction sees the exact
damage and every added pixel remains attributable to a hypothesis.

Reconstruction JSON contains:

    version
    cycle
    original_hypothesis
    diagnosis
    hypotheses
    accepted_hypothesis_ids

Each hypothesis records bridge geometry, original/reconstructed topology,
recognition evidence, acceptance score, added-pixel count, and debug paths.
Debug artifacts are written under:

    scribetrace/reconstruction/

The current implementation intentionally supports only one bridge per
hypothesis. Multi-repair search, loop-specific curved reconstruction, border
continuation, and learned reconstruction ranking are future additive stages.

Purpose
-------
ScribeTrace converts N02 binary ink masks into deterministic vector evidence.
It transforms raster ink into topology, ordered paths, geometric landmarks, a
stable numeric feature vector, and a symbolic path sequence.

It stops before stroke matching and OCR: vectors are evidence, not characters.

Processing Flow
---------------
1. Use a readable analysis_mask_crop_path as exact white-on-black ink.
2. If that mask is unavailable, convert the visual crop with inverse Otsu.
3. Extract deterministic 8-connected components.
4. Reject components below minimum_ink_pixels and rebuild a clean mask.
5. Stop with completed_limited if the accepted component limit is exceeded.
6. Thin the clean mask with internal Zhang-Suen thinning.
7. Build a pruned 8-neighbor graph that preserves genuine diagonals while
   removing redundant corner diagonals.
8. Classify topology and contract connected junction pixels into logical nodes.
9. Traverse every graph edge, including junction-to-junction paths and loops.
10. Detect enclosed background holes inside accepted ink components.
11. Match visual holes against closed skeleton paths when their boxes overlap.
12. Conservatively merge only unambiguous directional terminal spurs.
13. Treat each ordered path as coordinate signals x(t) and y(t).
14. Extract global boundaries, local extrema, movement, turn, and entropy data.
15. Encode deterministic numeric features and symbolic path tokens.
16. Save compact JSON evidence plus component, skeleton, graph, path, and
    landmark debug images.

Input Contract
--------------
The preferred input is the exact N02 mask:

    analysis_mask_crop_path

N05 exposes it to the expert as:

    scribetrace_mask_crop_path

The mask is exact target-layer evidence with white ink on black. A missing or
unreadable mask falls back to scribetrace_visual_crop_path and then crop_path.
Visual fallbacks are treated as dark ink on a light background.

Interface
---------

    expert.py
    TraceInput
    TraceSettings
    run_scribetrace(trace_input, settings=None)
    get_expert_manifest(settings=None)
    recognize(crop_path, context=None, settings=None)

The last two functions preserve the shared N05 package interface while routing
work into run_scribetrace().

Module Structure
----------------
expert.py is now the stable public facade and pipeline orchestrator. Existing
imports from scribetrace.expert remain valid while implementation details live
in focused modules:

    trace_common.py
        Shared constants, ordering keys, filename sanitization, and edge keys.

    trace_settings.py
        TraceSettings validation and normalization.

    trace_models.py
        Input, point, bounding-box, component, hole, path, landmark, result,
        and feature-vector records.

    trace_masks.py
        Mask source resolution, binarization, component filtering, visual-hole
        detection, and hole-to-loop matching.

    trace_skeleton.py
        Zhang-Suen thinning, skeleton point extraction, graph construction,
        crossing-number topology, and logical junction clusters.

    trace_segmentation.py
        Character-boundary proposals combining projection valleys, connected
        component attachment, exact TracePath edge lookup, virtual graph cuts,
        and coherent left/right vector-subgraph validation.

    trace_reconstruction.py
        Damage diagnosis, endpoint-tangent bridge hypotheses, copied-mask
        reconstruction, complete retracing, verification, ranking, and
        acceptance.

    trace_paths.py
        Complete edge traversal, deterministic loops, terminal-spur merging,
        coordinate signals, and landmark extraction.

    trace_features.py
        The deterministic 104-feature encoder and symbolic sequence builder.

    trace_debug.py
        Component, skeleton, graph, path, and landmark debug rendering.

    trace_inference.py
        Cached Random Forest loading, schema alignment, and top-k letter
        candidates.

This split is organizational only. A real-mask regression snapshot confirmed
that components, paths, landmarks, metrics, and all 104 feature values remain
identical to the monolithic implementation.

Result Contract
---------------
status values:

    disabled
    completed
    completed_limited
    failed

completed_limited is attempted expert evidence. It contains component summaries
but intentionally skips skeleton graph and path extraction. Its reason is:

    component_limit_exceeded

Source metadata records:

    source_path
    source_type
    fallback_used
    requested_threshold_mode
    threshold_mode

Filtering metrics record raw, accepted, and rejected-small component counts.
Topology metrics record point, edge, endpoint, junction-cluster, isolated-point,
loop, short-path, and directional-merge counts.

Ink-Hole Evidence
-----------------
Skeleton loops and visible holes are related but not identical evidence. A
thinned centerline can break or create graph structures near thick junctions,
so ScribeTrace also measures enclosed background directly from the cleaned ink
mask.

Each InkHole records:

    hole_id
    component_id
    point_count
    bounding-box area and center

Hole-to-path matching uses bounding-box overlap and records matched and
unmatched counts. This gives the classifier counter/loop evidence even when
skeleton topology alone is imperfect.

JSON Output
-----------
Enabled traces save their compact machine-readable result automatically:

    temp_processing/<document_id>/n05_handwritten_ocr/scribetrace/metadata/
        <stable_unit_id>_scribetrace.json

The JSON contains components, topology metrics, paths, landmarks, feature names,
the numeric vector, and the symbolic sequence. Full per-pixel arrays stay out of
this normal result file; the debug images remain the visual inspection format.

Set save_json=false only for callers that explicitly do not want this artifact.
Standalone execution prints a short status/count summary and the JSON path
instead of dumping the complete result dictionary into the terminal.

Path Evidence
-------------
Each TracePath records:

    deterministic path_id
    point_count
    geometric length
    start/end topology
    junction cluster IDs
    is_closed
    is_short
    merged_from_path_ids

Closed loops use a stable lexicographic anchor and matching start/end point.
Isolated dots remain graph evidence and do not become invented paths.

Landmark Evidence
-----------------
Each ordered path is treated as two discrete signals:

    x(t): horizontal motion
    y(t): vertical motion

Global landmarks record the path start, end, left, right, top, and bottom.
Local extrema record:

    local_top_peak
    local_bottom_valley
    local_left_turn
    local_right_turn

Local extrema must satisfy configurable prominence and spacing requirements.
This suppresses pixel jitter while preserving meaningful Armenian stroke shape.

Vector Evidence
---------------
TraceFeatureEncoder produces:

    vector
        Stable numeric geometry features sorted by feature name.

    feature_names
        The exact ordered schema corresponding to vector positions.

    sequence
        Ordered path and landmark tokens for sequence models.

    sequence_string
        Human-readable version of the symbolic sequence.

Numeric features currently include component, path, endpoint, junction, loop,
short-path, landmark, path-length, ink-density, and hole evidence. Directional
features additionally describe:

    horizontal and vertical direction changes
    clockwise and counterclockwise turns
    up, down, left, right, and diagonal step counts and ratios
    net and absolute path displacement
    path straightness
    direction entropy

The current encoder emits 104 numeric features. Feature names are alphabetically
ordered before vector serialization, making one implementation deterministic
across repeated runs.

The feature_names array is the authoritative model contract. Adding, removing,
or renaming any feature requires a new dataset export and model retraining.
Models must never consume a vector only because its schema_version string
matches; they must also verify the exact ordered feature_names list.

Example symbolic shape:

    P0 S TP BV TP E | P1:LOOP S GL GT GR GB E

Debug Output
------------
When save_debug=true, files are owned by N05 instead of being written beside
the N02 source mask:

    temp_processing/<document_id>/n05_handwritten_ocr/scribetrace/debug/
        <stable_unit_id>_components_debug.png
        <stable_unit_id>_skeleton_debug.png
        <stable_unit_id>_skeleton_graph_debug.png
        <stable_unit_id>_trace_paths_debug.png
        <stable_unit_id>_landmarks_debug.png

Debug identifiers are sanitized before filenames are created.
Set debug_draw_labels=true to draw component IDs, path IDs, and landmark
abbreviations. It defaults to false so the geometry remains readable.

Current Boundary
----------------
ScribeTrace does not yet infer physical pen stroke order, compare against
Armenian pattern libraries, rank characters, or return final OCR text. Those
stages should consume its vector and sequence evidence rather than re-reading
thick raw ink.

Directional names describe deterministic graph traversal, not recovered pen
motion. They are useful geometric signals, but should not be interpreted as the
writer's real stroke order.

Pipeline Status
---------------
The ScribeTrace implementation can run standalone, but its N05 expert flag is
currently disabled in settings.json. The integrated N05 pipeline therefore
continues to emit placeholder expert results.

Tests
-----

    .venv/bin/python -m unittest \
        tests.N05handwritten_ocr.scribetrace.test_expert

Scrilog Expert
==============

Purpose
-------
Scrilog is the symbolic geometry expert in the Scribe family. It receives the
clearest active ScribeTrace feature vector and deduces which Armenian letters
are structurally compatible with that evidence.

Scrilog does not inspect pixels, modify masks, generate reconstruction
hypotheses, or consume ANTAR probabilities. It independently evaluates every
supported letter class and returns its own explainable top-five candidates.

The name combines ScribeTrace with Prolog-style logical reasoning:

    ScribeTrace + logic = Scrilog

Pipeline Position
-----------------
Scrilog runs after theoretical reconstruction has finished retracing and
selecting the strongest accepted repair:

    input mask
        -> initial ScribeTrace analysis
        -> condition detection and defense routing
        -> theoretical reconstruction hypotheses
        -> complete retracing of every evaluated hypothesis
        -> strongest accepted reconstruction selected
        -> selected feature vector promoted as active evidence
        -> Scrilog top-five
        -> ANTAR top-five

Scrilog and ANTAR receive the same active 104-feature vector but reason about
it differently:

    ANTAR
        Learns statistical decision boundaries with a Random Forest.

    Scrilog
        Applies explicit Armenian structural constraints and weighted evidence.

If no reconstruction is accepted, the original ScribeTrace feature vector
remains active and is used by both experts.

Core Principle
--------------
Scrilog asks:

    Given the observed topology, which letters remain structurally possible?

It reasons from facts such as:

    endpoint count
    component count
    junction-cluster count
    closed-loop and visible-hole counts
    path count and path-length distribution
    horizontal, vertical, and diagonal movement ratios
    clockwise and counterclockwise turn evidence
    bounding-box proportions and fill ratio
    endpoint distribution across the bounding box
    landmark peaks, valleys, turns, and boundaries
    ink balance across halves, quadrants, center, and edges

Scrilog evaluates all classes. It does not begin from ANTAR's shortlist. This
preserves an independent failure pattern, which is essential for later
mixture-of-experts agreement.

Expected Topology Contract
--------------------------
Scrilog Lab stores human-reviewed canonical geometry in:

    datasets/scrilog/scrilog_annotations.json

Each record uses `expected_signature`. ScribeTrace emits the matching observed
geometry under `metrics.scrilog_observation`; reconstructed hypotheses also
carry `scrilog_observation`, and reconstruction exposes the selected value as
`selected_scrilog_observation`.

The v2 contract keeps visual ink holes separate from closed skeleton loops and
records endpoints and logical junction clusters by glyph-relative quadrant.
Quadrant totals must equal their corresponding global counts. Border contacts
are objective mask-edge measurements. Loop, branch, wide, and tall families
are derived rather than independently entered where possible.

The discarded ascender, descender, left-exit, and right-exit flags are not part
of this contract. They depended on unstable semantic guesses from endpoint
half-ratios and could not provide reliable expected-versus-observed deltas.

Scrististics can therefore compare:

    expected_signature - selected_scrilog_observation

per true letter, damage condition, writer, and reconstruction decision without
changing ANTAR's frozen 104-feature model schema.

Evidence Importance Policy
--------------------------
The annotation document stores one global `evidence_policy` rather than
duplicating weights in every letter record. Each expected field has a high,
medium, or low importance tier, a numeric weight, and an exact `observed_path`
inside `reconstruction.selected_scrilog_observation`.

High evidence covers stable structural identity such as holes, closed loops,
endpoint count, junction clusters, components, and coarse shape. Medium
evidence covers paths, isolated/short fragments, and objective border contact.
Low evidence covers endpoint and junction quadrants. Damage-aware reliability
is applied separately later; importance must never be mistaken for confidence
in one damaged observation.

Reasoning Model
---------------
Each Armenian class will have a versioned rule profile containing several
kinds of evidence.

Required evidence:
    Structure normally necessary for a class. Missing required evidence causes
    a strong penalty, but only becomes a hard rejection when the feature is
    considered reliable under the current damage condition.

Forbidden evidence:
    Structure considered incompatible with a class, such as an impossible loop
    or junction arrangement. Reliable contradictions can eliminate a class.

Expected ranges:
    Numeric intervals for continuous or variable evidence such as aspect ratio,
    path length, fill ratio, centroid position, and directional balance.

Supporting evidence:
    Features that strengthen a class without being mandatory.

Relational evidence:
    Rules comparing multiple measurements, for example:

        endpoint_count >= 2
        and bottom_half_ink > top_half_ink
        and vertical_spread > horizontal_spread

Damage-aware reliability:
    A missing endpoint after light_cut is less trustworthy than the same missing
    endpoint on a clean trace. Scrilog therefore separates feature value from
    feature reliability.

Rule Strengths
--------------
Rules use three strengths:

    hard contradiction
        Eliminates a class only when the evidence is structurally impossible
        and reliable.

    soft contradiction
        Reduces a class score while preserving it as a possible candidate.

    supporting evidence
        Raises a class score when expected geometry is observed.

This distinction prevents damaged handwriting from being rejected merely
because reconstruction failed to recover one endpoint, tail, loop, or path.

Candidate Scoring
-----------------
Scrilog will calculate one independent score for every class. A conceptual
score is:

    class_score =
        base_prior
        + supporting_evidence
        - soft_contradictions
        - range_distance_penalties
        - hard_contradiction_penalties

Every contribution is weighted by feature reliability. Scores are normalized
only after all classes have been evaluated. Deterministic tie-breaking uses the
numeric class identifier so repeated runs produce identical rankings.

Hard elimination must remain rare. A class removed by a hard rule must record
the exact rule, observed value, expected contract, and reliability that caused
the rejection.

Input Contract
--------------
The initial Scrilog input should contain:

    unit_id
    feature_names
    vector
    sequence
    sequence_string
    active_feature_source
    selected_reconstruction_id
    condition_verdict
    reconstruction metadata

The feature contract comes from:

    scribetrace/trace_features.py
    scribetrace/trace_models.py

Scrilog must map features by name rather than assuming an undocumented numeric
position. It must reject incompatible feature schemas loudly.

The active_feature_source is expected to be either:

    original
    reconstructed

Output Contract
---------------
Scrilog should return a JSON-safe result shaped like:

    {
      "expert": "scrilog",
      "status": "completed",
      "unit_id": "...",
      "feature_source": "reconstructed",
      "schema_version": "scrilog_rules_v1",
      "evaluated_class_count": 78,
      "top_candidates": [
        {
          "rank": 1,
          "class_id": 12,
          "label": "...",
          "score": 0.87,
          "supporting_rules": ["endpoint_range_matches"],
          "soft_contradictions": [],
          "hard_contradictions": [],
          "feature_evidence": []
        }
      ],
      "eliminated_class_count": 31,
      "diagnostics": {}
    }

The top candidate list contains at most five classes. Full per-class reasoning
may be retained in diagnostics or debug output, but normal N05 metadata should
remain compact.

Explainability Contract
-----------------------
Every score adjustment must be traceable. A rule result should record:

    rule_id
    feature names
    observed values
    expected values or ranges
    reliability
    score adjustment
    explanation

Example:

    {
      "rule_id": "endpoint_range_matches",
      "features": ["endpoint_count"],
      "observed": 3,
      "expected": {"minimum": 2, "maximum": 4},
      "reliability": 0.92,
      "adjustment": 0.18,
      "explanation": "Observed endpoint count supports this class."
    }

Rule Knowledge Base
-------------------
The rule knowledge base should eventually combine two sources.

Empirical profiles:
    Robust per-class distributions derived from clean Matenadata traces. Median,
    percentile ranges, and occurrence rates provide initial numeric envelopes.

Authored Armenian rules:
    Human-reviewed constraints describing decisive loops, tails, endpoints,
    junctions, turns, and directional structures.

Empirical values must be generated only from training partitions. Validation
and test samples must never contribute to class profiles.

One perfect letter is useful as an explanation prototype, but it cannot define
an entire handwritten class. Each class should support multiple valid style
profiles or robust distributions.

Relationship to Scrististics
----------------------------
Scrilog and Scrististics have separate responsibilities.

    Scrilog
        Produces an independent top-five by applying structural rules to the
        active vector.

    Scrististics
        Acts as shared statistical memory between pixel-space and vector-space
        experts. It explains known feature-loss patterns and cross-modal
        disagreements; it is not another voting expert.

Scrilog states what is structurally possible. Scrististics later explains why
expected structure may be missing.

Safety Rules
------------
Scrilog must:

    never alter the source or reconstructed mask
    never alter the active ScribeTrace vector
    never consume ANTAR probabilities when generating its own ranking
    never eliminate a class from unreliable evidence alone
    never silently accept an incompatible feature schema
    remain deterministic for identical inputs and rule versions
    preserve complete reasoning for debugging

Module Structure
----------------
Scrilog is split by responsibility while preserving scrilog.py as its public
facade and direct command-line entrypoint:

    scrilog.py
        Backward-compatible public imports and CLI handoff.

    constants.py
        Version, bounded-pass defaults, and shared identifiers.

    utils.py
        Defensive JSON conversion and dictionary helpers.

    facts.py
        Hashable facts and the indexed symbolic fact store.

    signature.py
        Normalized symbolic geometry record.

    signature_builder.py
        Signature construction orchestration.

    signature_extractors.py
        Schema-tolerant extraction of real ScribeTrace fields.

    results.py
        Candidate effects and JSON-safe final result records.

    engine.py
        Bounded deterministic forward-chaining engine.

    rules.py
        Generic structural-family and reconstruction-safety rules.

    profiles.py
        Armenian class profiles and candidate-effect evaluation.

    parser.py
        Signature, facts, rules, profiles, and explanation orchestration.

    io.py
        JSON file handling and command-line implementation.

    __init__.py
        Package-level public API.

Future scoring and versioned settings can be added as focused modules without
growing the facade again.

Development Sequence
--------------------
1. Define the immutable input and output contracts.
2. Load and validate the ScribeTrace feature schema.
3. Represent feature facts by name.
4. Implement deterministic numeric range rules.
5. Add hard, soft, and supporting rule strengths.
6. Score all classes and return an explainable top-five.
7. Generate initial empirical class profiles from training-only clean traces.
8. Add reviewed Armenian structural rules incrementally.
9. Compare Scrilog against ANTAR on identical held-out samples.
10. Measure top-one, top-five, disagreement, elimination accuracy, and runtime.

Success Criteria
----------------
Scrilog is successful when it provides useful independent evidence rather than
merely copying ANTAR's ranking. Evaluation should measure:

    clean top-one and top-five accuracy
    degraded top-one and top-five accuracy
    percentage of true classes incorrectly hard-eliminated
    ANTAR/Scrilog agreement and disagreement
    cases where one expert rescues the other
    per-rule support and contradiction frequency
    deterministic repeatability
    inference time per glyph

The most important safety metric is false elimination. A useful symbolic expert
may rank imperfectly, but it must not repeatedly remove the correct Armenian
letter because damaged evidence was treated as certain.

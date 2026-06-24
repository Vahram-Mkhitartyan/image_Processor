N05 Assembly
============

Purpose
-------
N05 Assembly is the first decision-surface layer for the handwriting expert
stack.

It does not perform final OCR yet.

It takes the outputs of the existing N05 components and places them into stable
JSON contracts that later scoring, beam search, and adaptive formula selection
can consume.

Current v0.1 Flow
-----------------
Input:

    handwritten_text_units

Each unit may contain:

- character_unit_proposal
- word_level_ocr output
- scribetrace output
- character_detector output
- future ScriLog / ScriStatistics evidence

Assembly builds:

1. Segmentation matrix
   Candidate ways to split or preserve a text unit.

2. Letter matrix
   Position-level candidate letters collected from available experts.

3. Assembly summary
   How much evidence exists, how many rows are populated, and whether final
   decision logic is active.

Current v0.2 Additions
----------------------
The assembly layer now also records the first three execution hooks needed by
the future giant:

1. Word OCR split hints
   `prediction.split_line_candidates` are copied into each segmentation matrix
   entry as `word_ocr_split_evidence`.

   Strong split hints are now also allowed to become a real candidate
   segmentation path:

       wocr_p0_boundaries

   This path uses the Word OCR boundary head to propose segment boxes. Existing
   character-unit-proposer paths are not removed; they are scored against the
   same Word OCR boundary evidence so the matrix can compare both sources.

   Bridge output from the current word model is still a soft count
   (`predicted_bridge_count`), not bridge coordinates. Assembly stores it as
   evidence, but does not draw bridge-aware split boxes yet.

2. Selected-path segment crops
   The first ranked segmentation path is materialized into:

       assembly/segments/visual/
       assembly/segments/mask/

   Candidate paths remain available, but only the selected path is cropped in
   v0.2 to avoid candidate explosion.

3. Segment condition evidence
   Each materialized segment receives a conservative image-only condition
   verdict and routing advice. This is a placeholder for the future
   segment-level ScribeTrace-backed condition classifier.

4. Matrix envelope
   The old top-level `segmentation_matrix` and `letter_matrix` fields are still
   preserved for compatibility, but the same evidence is also wrapped in a
   future-ready `matrices` block:

       matrices.segmentation
       matrices.condition
       matrices.reconstruction
       matrices.letter_evidence
       matrices.sequence

   This lets later versions add new decision layers without breaking the
   existing N05 output shape.

5. Expert job queue
   The selected segment path now produces queued expert jobs. These jobs do not
   execute yet; they describe what the future assembler should run per segment:

       condition -> reconstruction if needed -> scribetrace -> scrilog
       -> scrististics -> character_detector

6. Case fingerprint and correctness history placeholder
   `case_fingerprint` records the document-level situation: word OCR evidence,
   condition mix, segment count, and populated letter rows. `correctness_history`
   is reserved for the future learner that will remember which formulas worked
   for which kinds of cases.

7. Manual expert probes
   During development, individual segment masks can be sent through the four
   current evidence families:

       word_level_ocr
       scribetrace
       scrilog / scrististics
       character_detector

   Probe outputs are written under:

       temp_processing/<document_id>/n05_handwritten_ocr/manual_expert_probe/

   These JSONs are not the final decision matrix yet. They are the evidence
   packets used to inspect which expert is structurally reliable for a segment.

Important Boundaries
--------------------
This layer DOES:

- normalize segmentation hypotheses
- preserve source paths and bboxes
- normalize letter candidates
- preserve expert provenance
- expose empty rows when evidence is missing
- make debugging easier
- compare proposer paths against Word OCR split-line evidence
- generate a Word-OCR-boundary segmentation path when the boundary head is
  confident enough
- materialize selected-path segment crops
- attach first-pass damage/condition evidence
- expose a multi-matrix envelope for future decision layers
- queue future per-segment expert work
- reserve correctness-history metadata
- preserve enough paths and artifact links for manual four-expert probes

This layer DOES NOT:

- pick the final word
- decide final letter identity
- run expensive experts
- repair crops automatically
- replace ScribeTrace, ScriLog, ScriStatistics, CNN, or word OCR
- learn the final formula yet
- blindly trust the loudest model confidence

Why This Exists
---------------
The final N05 reasoning engine should not be a pile of unrelated expert JSONs.
It needs a shared surface:

    segmentation paths x letter candidates x expert evidence

That shared surface is what the future decision matrix and correctness-history
AI will learn from.

Future Versions
---------------
Planned upgrades:

- beam search over segmentation paths
- crop candidate queue/deferred candidate handling
- segment-level ScribeTrace
- segment-level CNN
- ScriLog/ScriStatistics geometry evidence per cell
- final formula scoring
- correctness-history training data export
- expert reliability weighting based on clean history and case fingerprints

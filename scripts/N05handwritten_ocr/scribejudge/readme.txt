ScribeJudge
===========

Purpose
-------
ScribeJudge is the learned/referee layer for N05. It does not replace
ScribeTrace, ScriLog, ScriStatistics, CNN, word OCR, or ScribeTrain. It learns
how to judge their disagreements.

Current v0.1 role
-----------------
The current implementation is an advisory overlay:

    N05 experts -> combined_expert_output -> decision_matrix -> scribejudge_overlay

It reads:
- candidate scores and source agreement
- backup letters per position
- conflicts in the letter matrix
- historical confusion reports
- fake-high-confidence / fake-low-confidence patterns

It writes:
- scribejudge_overlay rows beside the decision matrix
- per-position confusion-risk records
- advisory flags, not final truth
- JSONL-ready rows for later training

Important principle
-------------------
ScribeJudge should not become another OCR. It should become the judge that knows
which expert tends to be wrong in which situation.

Confusion memory
----------------
The first memory sources are:

    reports/scribetrace_random_forest_v4_0/confusion_pairs.json
    reports/glyph_classifier_v0_2_aristotel/confusion_matrix.csv

Each predicted -> true pair becomes a risk signal. Example idea:

    selected candidate: Յ
    backup candidate: Է
    confusion memory says Յ is often actually Է
    ScribeJudge marks that position as suspicious high confidence

Dataset builder
---------------
Build audit/training rows from existing N05 maps:

    .venv/bin/python scripts/Cyber_Lin_Kuei_Assembly/scribejudge/scribejudge_dataset_builder.py \
        --input temp_processing/test_4/n05_handwritten_ocr/metadata/test_4_handwritten_text_map.json \
        --output datasets/scribejudge/test_4_scribejudge_rows.jsonl

Baseline report:

    .venv/bin/python scripts/Cyber_Lin_Kuei_Assembly/scribejudge/scribejudge_trainer.py \
        --dataset-jsonl datasets/scribejudge/test_4_scribejudge_rows.jsonl

Future training
---------------
The real training loop should:

1. Generate synthetic words with known truth.
2. Run the full N05 pipeline.
3. Attach truth_text to each ScribeJudge row.
4. Train a meta-model to predict whether the decision matrix is correct and
   which backup should be elevated.
5. Feed the calibrated judge back into the final N05 output.

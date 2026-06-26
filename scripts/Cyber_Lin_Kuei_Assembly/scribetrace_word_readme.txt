ScribeTrace Word v0.1
=====================

Purpose
-------
ScribeTrace Word is an experiment that tests whether raw ScribeTrace geometry
can become a better word splitter than pixel word OCR.

It is not replacing the current word OCR yet.

Current Flow
------------

    synthetic word generator
    -> raw ScribeTrace on the whole word image
    -> feature vector + symbolic trace sequence
    -> RandomForest heads

Targets
-------
The exported JSONL stores:

    text
    token_ids
    padded_token_ids
    length
    bridge_count
    split_x_positions
    boundary_bins
    ScribeTrace feature vector
    ScribeTrace sequence_string

The v0.1 model trains three simple heads:

    length_model
    sequence_model
    boundary_model

The most important early metric is `boundary.f1`. Recognition can be weak while
the splitter is still useful.

Important Boundaries
--------------------
This experiment uses raw ScribeTrace only:

    no theoretical reconstruction
    no ANTAR
    no ScriLog
    no Scrististics

Those tools should be applied later after ScribeTrace Word proposes candidate
letter spans.

Smoke Test
----------

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_trainer.py \
        --mode export-train \
        --limit 80

Outputs:

    datasets/scribetrace_word_v0_1/
    models/scribetrace_word_v0_1/
    reports/scribetrace_word_v0_1/

Larger Test
-----------

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_trainer.py \
        --mode export-train \
        --limit 2000

Full configured export/training:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_trainer.py \
        --mode export-train

Decision Rule
-------------
If ScribeTrace Word has worse recognition but better boundary scores than pixel
word OCR, keep both:

    pixel word OCR = text/sequence prior
    ScribeTrace Word = splitter/vector-span proposer

If it wins both recognition and boundary quality, promote it toward primary
word-level evidence.

## ScribeTrace Word Sequence Trainer v0.1

File: `scribetrace_word_sequence_trainer.py`

This is the splitter-first PyTorch lane for the ScribeTrace word experiment. Unlike the RandomForest baseline, it does not compress the whole word into one global vector. It renders a synthetic word, thresholds it into ink, skeletonizes it with ScribeTrace's Zhang-Suen skeletonizer, bins the word left-to-right, and extracts local topology features per bin.

The current targets are:
- `boundary`: whether an x-bin is close to a true letter split.
- `length`: the number of letters in the synthetic word.

This is intentionally not a final OCR model. Its job is to learn where the word can be cut so downstream experts can run on better character candidates.

Smoke command:

```bash
.venv/bin/python -u scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_sequence_trainer.py --limit 64 --epochs 2
```

Configured training command:

```bash
.venv/bin/python -u scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_sequence_trainer.py
```

Outputs:
- `models/scribetrace_word_sequence_v0_1/scribetrace_word_sequence_v0_1.pt`
- `reports/scribetrace_word_sequence_v0_1/training_report.json`

The settings live in `scribetrace_word_sequence_settings.json`. The default `num_workers` is `0` because some local/sandboxed environments block PyTorch multiprocessing sockets. On a normal training machine, this can be raised.

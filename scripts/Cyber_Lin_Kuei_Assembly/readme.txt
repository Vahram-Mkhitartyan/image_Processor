Cyber Lin Kuei Assembly
=======================

Purpose
-------
Training, evaluation, and preserved ML experiments live here. Runtime node code
does not belong in this folder.

Files
-----
train_minos_classifier.py:
    Trains Minos on the four visual classes.

evaluate_minos_v2.py:
    Evaluates saved Minos models and threshold behavior.

glyph_classifier.py:
    Glyph-model training/experiment code for the N05 character_detector CNN.
    It can train clean Matenadata glyphs alone or add lightweight Aristotel
    damage variants in memory.

evaluate_glyph_classifier.py:
    Evaluates saved glyph classifier behavior.

word_level_ocr_trainer.py:
    Trains the N05 word-level OCR expert. It synthesizes Armenian word images
    from Matenadata glyphs plus the Armenian word-frequency corpus, then trains
    a compact CRNN/CTC recognizer.

scribetrace_word_trainer.py:
    Tests raw ScribeTrace as a word-level splitter/recognizer. It renders the
    same synthetic words as word_level_ocr_trainer.py, traces the whole word,
    exports ScribeTrace feature vectors, and trains simple RandomForest heads
    for length, token sequence, and boundary-bin prediction.

word_level_ocr_settings.json:
    Owns the synthetic word dataset size, rendering controls, CRNN training
    hyperparameters, and output locations.

scribetrace_word_settings.json:
    Owns the raw ScribeTrace-word export size, trace settings, boundary-bin
    target size, and RandomForest baseline settings.

scribetrace_random_forest.py:
    Exports deterministic ScribeTrace geometry vectors from Matenadata and
    trains a Random Forest glyph-classification baseline.

scribetrace_random_forest_settings.json:
    Owns dataset/module routing, ScribeTrace extraction controls, split ratios,
    output folders, and Random Forest hyperparameters.

tests/Cyber_Lin_Kuei_Assembly/scribemap_v1_classifier_test.py:
    Preserved old ScribeMap/classifier experiment. It is not part of the active
    runtime pipeline.

Minos Classes
-------------

    mixed
    handwriting_only
    printed_only
    empty_or_noise

Runtime Status
--------------
Minos is active in N03 through:

    models/minos_v2_0_best.keras

N03 converts the three sigmoid outputs printed_present, handwriting_present, and
noise into the four route classes plus review.

Dataset
-------
The Minos dataset lives under classifier_dataset_presence/ and is preserved by
main.py clean. Models live under models/ and are also preserved.

Command
-------

    .venv/bin/python scripts/main.py train

Run the ScribeTrace Random Forest smoke export:

    .venv/bin/python \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_random_forest.py \
        --mode export --limit-per-class 1

Run the configured export and training baseline:

    .venv/bin/python \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_random_forest.py \
        --mode export-train

Train an already exported full JSONL dataset:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_random_forest.py \
        --mode train \
        --dataset-jsonl datasets/scribetrace_random_forest_v0_2/scribetrace_rf_full.jsonl \
        2>&1 | tee reports/scribetrace_random_forest_v0_2/full_train_from_export.log

The trainer reads JSONL into a compact float32 feature matrix and splits sample
indexes rather than retaining duplicate Python dictionaries. The configured
forest also caps depth, leaves per tree, and worker count so full training fits
an 8 GB machine. The leaf cap is especially important for 78-class trees,
because every tree node stores one probability value per class.

ScribeTrace Random Forest v0.2
------------------------------
v0.2 uses the 61-feature ScribeTrace vector, including visible ink holes,
direction changes, turn orientation, movement ratios, displacement,
straightness, and direction entropy.

The 7,800-sample subset produced:

    test top-1: 0.6077
    test top-5: 0.8436

The complete 70,060-sample export produced:

    validation top-1: 0.5784
    validation top-5: 0.8304
    test top-1: 0.5674
    test top-5: 0.8279

The full result is the authoritative benchmark. The smaller export uses the
first limited set of sorted images per class and is therefore not a random,
representative subset of the full class distribution. Its higher score should
not be interpreted as evidence that additional data reduces model quality.

Artifacts:

    models/scribetrace_random_forest_v0_2/
    reports/scribetrace_random_forest_v0_2/
    datasets/scribetrace_random_forest_v0_2/

Any ScribeTrace feature change requires a fresh JSONL export and retraining.
Inference should compare exact ordered feature_names against the model schema
before calling predict_proba().

The folder name is ridiculous in the correct way. Keep it.

Word-Level OCR v0.1
-------------------
The word-level OCR arena starts from synthetic words because we do not yet have
a large labeled real-word crop dataset. It renders real Matenadata glyphs into
word images, trains CTC sequence recognition, and writes artifacts to:

    models/word_level_ocr_v0_1/
    reports/word_level_ocr_v0_1/

Render a few synthetic debug words:

    .venv/bin/python \
        scripts/Cyber_Lin_Kuei_Assembly/word_level_ocr_trainer.py \
        --mode smoke --smoke-count 12

Quick overfit/debug training:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/word_level_ocr_trainer.py \
        --mode train --limit-batches 5

Configured training:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/word_level_ocr_trainer.py \
        --mode train

ScribeTrace Word v0.1
---------------------
This experiment asks a narrower question than word OCR:

    can raw ScribeTrace word geometry become a better splitter?

It uses the same synthetic word generator, but feeds whole-word ScribeTrace
vectors into RandomForest heads. Recognition is measured, but boundary F1 is
the key early score.

Smoke export and training:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_trainer.py \
        --mode export-train \
        --limit 80

Larger experiment:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_trainer.py \
        --mode export-train \
        --limit 2000

Configured experiment:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_trainer.py \
        --mode export-train

See:

    scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_readme.txt

Character CNN + Aristotel
-------------------------
The character-detector CNN can now use Aristotel as a robustness teacher. The
trainer keeps validation and test clean, but exposes extra damaged training
views virtually through the Dataset. It does not write multiplied damaged PNGs
to disk.

Smoke test:

    .venv/bin/python \
        scripts/Cyber_Lin_Kuei_Assembly/glyph_classifier.py \
        --model-name glyph_classifier_aristotel_smoke \
        --limit-per-class 10 \
        --epochs 1 \
        --batch-size 128 \
        --num-workers 0 \
        --use-aristotel \
        --aristotel-variants-per-sample 1 \
        --aristotel-recipes light_cut light_erosion threshold_failure light_blur

Full Matenadata training:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/glyph_classifier.py \
        --model-name glyph_classifier_v0_2_aristotel \
        --limit-per-class -1 \
        --epochs 20 \
        --batch-size 128 \
        --num-workers 2 \
        --use-aristotel \
        --aristotel-variants-per-sample 4 \
        --aristotel-recipes light_cut light_erosion threshold_failure light_blur

Outputs:

    models/glyph_classifier_v0_2_aristotel/
    reports/glyph_classifier_v0_2_aristotel/

New checkpoints declare:

    input_polarity_mode = normalize_black_ink_on_white

That lets runtime normalize N05 white-on-black masks before applying the CNN's
usual one-minus-grayscale tensor transform.

Aristotel
---------
Aristotel is the deterministic degradation teacher under `Aristotel/`.
Its default `stream` mode generates training damage in RAM and stores no
multiplied image dataset. See `Aristotel/readme.txt` for storage modes.

ScribeTrace Random Forest v4.0
------------------------------
The v4 training settings connect clean Matenadata glyphs to Aristotel's
deterministic in-memory degradation and then extract ScribeTrace geometry.
Generated images are temporary; the retained dataset is compressed JSONL.

Settings:

    scripts/Cyber_Lin_Kuei_Assembly/
        scribetrace_random_forest_v4_settings.json

Small end-to-end validation:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_random_forest.py \
        --settings scripts/Cyber_Lin_Kuei_Assembly/\
scribetrace_random_forest_v4_settings.json \
        --mode export-train \
        --limit-per-class 10

Full export and training:

    .venv/bin/python -u \
        scripts/Cyber_Lin_Kuei_Assembly/scribetrace_random_forest.py \
        --settings scripts/Cyber_Lin_Kuei_Assembly/\
scribetrace_random_forest_v4_settings.json \
        --mode export-train \
        --limit-per-class -1

Every clean glyph and all Aristotel variants derived from it share source_id.
Splitting happens by source_id inside each class, so related variants cannot
leak between train, validation, and test. Reports include separate clean and
degraded metrics.

The v4 exporter also creates a compact recovery JSONL containing damage,
quality, and expected-recovery labels. This is the contract for the later
reconstruction gate. Actual reconstruction-benefit labels should be generated
after the v4 recognizer exists and can compare original versus reconstructed
confidence honestly.

Runtime activation is explicit through:

    models/scribetrace_active_model.json

It remains pinned to v0.2.1 after training. Change `model_name` to
`scribetrace_random_forest_v4_0` only after reviewing the full clean and
degraded benchmark reports. If the selected installation is incomplete,
runtime safely falls back to the proven model.

### ScribeTrace Word Sequence Splitter

`scribetrace_word_sequence_trainer.py` is the PyTorch splitter-first version of the ScribeTrace word experiment. It renders synthetic Armenian words, extracts left-to-right ScribeTrace-style topology features, and trains boundary/length heads. Use it when the goal is better character split proposals rather than direct word OCR.

Smoke:

```bash
.venv/bin/python -u scripts/Cyber_Lin_Kuei_Assembly/scribetrace_word_sequence_trainer.py --limit 64 --epochs 2
```

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
    Glyph-model training/experiment code for the future N05 expert family.

evaluate_glyph_classifier.py:
    Evaluates saved glyph classifier behavior.

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

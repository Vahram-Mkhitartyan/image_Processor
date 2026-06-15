ARISTOTEL
=========

Purpose
-------
Aristotel creates supervised damage examples from clean Matenadata glyphs.
The default workflow is storage-safe: damaged images are generated in memory,
consumed, and discarded instead of becoming a permanent multiplied dataset.

Determinism
-----------
Each sample is identified by:

    base seed + source_id + recipe + epoch + variant

The identity is hashed into a stable sample_id and NumPy seed. Repeating the
same coordinates produces the same damaged pixels. Changing the epoch or
variant creates a new reproducible example.

The manifest also stores a recipe signature. Aristotel refuses to regenerate
an old record if that recipe's operations or settings have since changed.

Storage modes
-------------
stream
    Default. Generate samples in RAM and write no files. Training code should
    consume AristotelRunner.iter_samples().

manifest
    Write one compact manifest.jsonl containing source, recipe, seed, epoch,
    and variant metadata. No damaged PNG files are stored. A sample can be
    recreated later with AristotelRunner.regenerate().

preview
    Save only a bounded number of damaged PNG examples for visual inspection.

export
    Explicitly materialize all selected images and per-image metadata. Use
    only for frozen benchmarks or transfer to another machine.

Commands
--------
No-storage smoke run:

    .venv/bin/python scripts/Cyber_Lin_Kuei_Assembly/Aristotel/run_aristotel.py \
        --mode stream --limit 100

Compact reproducibility manifest:

    .venv/bin/python scripts/Cyber_Lin_Kuei_Assembly/Aristotel/run_aristotel.py \
        --mode manifest --limit 1000 --variants 2

Twenty visual examples:

    .venv/bin/python scripts/Cyber_Lin_Kuei_Assembly/Aristotel/run_aristotel.py \
        --mode preview --limit 100 --preview-count 20

Intentional full export:

    .venv/bin/python scripts/Cyber_Lin_Kuei_Assembly/Aristotel/run_aristotel.py \
        --mode export --limit 100

Training integration
--------------------
Use:

    for sample in runner.iter_samples(epoch=current_epoch, variants=1):
        image = sample.image
        label = sample.teacher_input.label

Once the training framework consumes the array, no damaged image needs to
remain on disk. Validation and test manifests should use fixed epochs and
variants so model comparisons remain stable.

ScribeTrace v4 uses this streaming contract through
`scribetrace_random_forest_v4_settings.json`. The exporter writes only
compressed feature and recovery JSONL files; it does not retain degraded
glyph images.

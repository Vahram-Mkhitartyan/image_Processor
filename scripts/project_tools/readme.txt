PROJECT TOOLS
=============

This package contains developer diagnostics exposed through scripts/main.py.
They inspect pipeline behavior without changing source images or model files.

COMPARE
-------

Run:

    python scripts/main.py compare PATH_A PATH_B

The targets may be JSON files, images, arbitrary files, or directories. JSON
comparison reports changed fields recursively. Image comparison reports shape,
pixel, and foreground-ink changes. Directory comparison matches files by their
relative paths. Reports are written to reports/comparisons/ by default.

Use --output PATH to choose a report destination. A CHANGED result is useful
information and does not make the command fail.

BENCHMARK
---------

Create the intentional ScribeTrace geometry baseline:

    python scripts/main.py benchmark --update

Verify current behavior against it:

    python scripts/main.py benchmark

The baseline uses one deterministic Matenadata image from each class. It stores
source hashes, trace status, the exact feature schema and vector hash, and the
ScriLog observation. Reconstruction and debug output are disabled so this tests
the stable geometry engine rather than model inference or artifact rendering.

Normal benchmark runs reuse the exact baseline files and fail when deterministic
geometry changes. Runtime changes are warnings only. Update the baseline only
after reviewing and intentionally accepting a ScribeTrace behavior change.

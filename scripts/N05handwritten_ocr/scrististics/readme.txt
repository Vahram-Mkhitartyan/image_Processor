SCRISTISTICS
============

Scrististics builds transparent per-letter topology statistics from ScribeTrace
observations. It supports two input modes.

AUTOMATIC EMPIRICAL PROFILES
----------------------------

No manually filled annotation file is required. The class folder name provides
the true label and ScribeTrace provides observed geometry:

    python scripts/N05handwritten_ocr/scrististics/scrististics_miner.py \
      --matenadata Matenadata \
      --out datasets/scrististics/empirical_profiles.json

Use --limit-per-class 10 for a quick experiment. The default, -1, processes the
full dataset. Reconstruction is enabled by default and the selected retraced
candidate supplies the observation. Use --without-reconstruction to profile raw
traces instead.

For each class the miner stores individual feature modes and common correlated
joint signatures. A joint prototype is always tied to a real source glyph, so
the profile cannot combine incompatible properties into an imaginary letter.
Up to three common variants are retained for alternate writing styles.

The old aspect-ratio definition of tall_shape is excluded from automatic mining.
Tall now requires measurable double-tail/full-span evidence, which ScribeTrace
does not yet expose directly.

JSON INPUT MODE
---------------

Existing annotation or observation JSON remains supported:

    python scripts/N05handwritten_ocr/scrististics/scrististics_miner.py \
      --input observations.json \
      --out datasets/scrististics/profile.json

When records contain both expected_signature and observed geometry, the miner
also builds mismatch distributions and an error watchlist.

SCRILOG BRIDGE
--------------

Scrilog reads the newest empirical_profiles*.json through:

    scripts/N05handwritten_ocr/scrilog/scrististics_adapter.py

The profile supplies probabilistic class compatibility only. It never changes
observed facts and never creates a hard rejection. This keeps the statistical
inductive layer separate from Scrilog's deterministic deductive rules.

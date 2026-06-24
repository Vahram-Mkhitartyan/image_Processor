Word-Level OCR Expert
=====================

Purpose
-------
This expert works on complete Armenian word or text-unit crops. It is separate
from ScribeTrace, ScriLog, Scrististics, and the character CNN: those reason
about character geometry, while this expert will reason about whole sequences.

Current State
-------------
The node now has its own preprocessing, language-asset layer, and PyTorch
CRNN/CTC runtime:

- `preprocessing.py` loads crops, normalizes polarity, thresholds them, preserves
  aspect ratio, and prepares CTC-style model inputs.
- `language_assets.py` loads Armenian corpus/frequency files extracted from the
  capstone HTR archive.
- `inference.py` produces the N05 JSON evidence contract and checks whether word
  OCR model weights are configured.
- `model_runtime.py` loads trained `.pt` checkpoints and performs greedy CTC
  decoding.
- `expert.py` exposes `get_expert_manifest()` and `recognize()`.

Training lives in:

    scripts/Cyber_Lin_Kuei_Assembly/word_level_ocr_trainer.py

The trainer synthesizes Armenian word crops from Matenadata glyphs and the local
word-frequency table. It uses a compact CNN + BiLSTM + CTC model.

Structural Evidence
-------------------
This expert is not only trying to guess the word text. It also learns structural
signals that can help the rest of N05 reason about segmentation:

- `decoded_length`: length of the greedy decoded word.
- `predicted_length`: auxiliary head prediction for word length.
- `predicted_bridge_count`: estimated number of synthetic letter joins.
- `split_line_candidates`: approximate x positions where letter boundaries may
  exist.

These fields make the expert useful even when the exact word is wrong. For
example, a crop can still say: "this looks like an 8-letter unit with 3 possible
bridges and split candidates around these x positions."

Synthetic Rendering Notes
-------------------------
The trainer uses Armenian-specific rendering assumptions:

- Glyphs are placed into three horizontal bands.
- No-tail letters occupy mostly the middle band.
- Upper-tail, lower-tail, and double-tail letters occupy larger vertical spans.
- Neighboring letters may overlap or receive small connection bridges.
- Small per-word glyph rotation and slant imitate italic handwriting angles.
- Mild blur, morphology, noise, and crop jitter imitate N02/N05 crops.

Language Assets
---------------
Local assets live in:

    datasets/word_level_ocr/

Files:

- `armenian_ctc_corpus.txt`
- `armenian_word_frequencies.tsv`

These came from `/home/vahram/Downloads/capstone-htr-main.zip`. They are useful
for CTC Word Beam Search, language priors, and later post-processing.

Interface
---------
    expert.py
    get_expert_manifest(settings=None)
    recognize(crop_path, context=None, settings=None)

CLI Smoke Test
--------------
    python scripts/N05handwritten_ocr/word_level_ocr/expert.py path/to/crop.png --out temp_processing/word_ocr_smoke.json

Optional language-prior debug candidates:

    python scripts/N05handwritten_ocr/word_level_ocr/expert.py path/to/crop.png --language-prior

Expected Result Before Weights
------------------------------
If no model weights are configured, the expert returns:

    status = "model_missing"

That is not a crash. It means the crop preprocessor and language assets are ready,
but the CRNN/CTC recognizer has not been plugged in yet.

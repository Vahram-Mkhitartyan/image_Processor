# N06 Linguistic Resolver

N06 is the Armenian linguistic reasoning layer after N05 perception. It does
not trust OCR as final truth. It collects linguistic evidence that can explain,
reject, or repair noisy Armenian word candidates.

## First Active Tool: Letter N-Gram Stats

The first implemented signal is `letter_ngram_stats`.

Purpose:

- score whether a word candidate looks Armenian-like at the letter-sequence level
- flag OCR gibberish before morphology spends energy on it
- preserve unknown but plausible Armenian-looking words

It trains from:

```text
datasets/word_level_ocr/armenian_word_frequencies.tsv
```

The score is a z-score against real Armenian corpus words, not a dictionary
membership check.

## N-Gram Candidate Repair

`ngram_candidate_repair` uses N05 backup letters to search for better Armenian
letter sequences. It does not invent arbitrary alphabet edits. It starts from
low-confidence character positions and tries one-letter, then two-letter
substitutions from the candidate matrix.

Example input shape:

```json
{
  "text": "Մկտտու",
  "character_candidates": [
    {
      "index": 2,
      "char": "տ",
      "confidence": 0.41,
      "candidates": [
        {"char": "տ", "confidence": 0.41},
        {"char": "ր", "confidence": 0.36}
      ]
    }
  ]
}
```

Example repair:

```text
մկտտու -> մկրտու
z-score improves from -2.905 to -0.860
```

This is the first place where N06 starts arguing with N05 intelligently:

```text
The visual top-1 is ugly Armenian.
The visual backup letter creates a much healthier Armenian sequence.
Raise the repaired candidate.
```

## Current CLI

```bash
.venv/bin/python scripts/N06_linguistic_resolver/resolver_orchestrator.py \
  --words Մկրտչյան Մկտտու qqqxx \
  --output-json temp_processing/n06_linguistic_resolver/ngram_smoke.json
```

Output records keep `trusted_as_final=false`.

## Planned Axes

- structure: roots, prefixes, suffixes, postfixes, multiple roots, hodakap
- belonging: noun, verb, adjective, numeral, pronoun, linking/serving words, etc.
- context: synonyms, antonyms, ambiguity, similar words, document domain

N06 should answer: can this noisy candidate be explained as a valid Armenian
construction?

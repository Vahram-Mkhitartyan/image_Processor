TESTS
=====

All automated regression tests and manual integration checks live here,
separate from production node and training code.

Folders mirror the production structure:

    N00_file_preparation/
    N02_crop_refiner/
    N04_printed_ocr/
    N05handwritten_ocr/
    Cyber_Lin_Kuei_Assembly/

Run the automated suite:

    .venv/bin/python -m unittest discover -s tests -p "test_*.py"

Some files are manual integration tools requiring existing pipeline artifacts
or external OCR installations. Those files retain direct execution entrypoints
but are still counted under Tests by the sacred-lines protocol.

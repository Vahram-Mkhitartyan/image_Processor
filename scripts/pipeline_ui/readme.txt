Pipeline Control Room
=====================

Purpose
-------
This package provides the dependency-free local browser UI launched by:

    .venv/bin/python main.py ui

The server binds only to 127.0.0.1. It runs existing main.py commands as
subprocesses, streams their combined terminal output, reports per-document node
completion, and safely serves image artifacts from inside the project.

Files
-----
server.py:
    Local HTTP server, command process manager, document telemetry, and safe
    artifact catalog.

static/index.html:
    Dashboard structure.

static/styles.css:
    Visual system for the control room.

static/app.js:
    Phase execution, live log polling, document selection, and image tracing.

Current Debug Features
----------------------
The artifact gallery can inspect node outputs without leaving the browser.

For N02 crop artifacts, the UI also asks the server for crop context. When the
metadata can be matched, the detail view shows the original document with the
crop bbox overlay beside the selected crop. This is especially useful when a
refined crop contains stacked words, red corrections, or suspicious partial
letters.

The same artifact server path is used by manual N05 expert probes, so copied
artifact URLs can be resolved back into project-relative files.

Environment
-----------
OCR_PIPELINE_UI_PORT:
    Preferred local port. Defaults to 8765 and advances if occupied.

OCR_PIPELINE_UI_NO_BROWSER=1:
    Start the server without automatically opening the default browser.

Safety
------
Only allow-listed main.py commands can run. One command runs at a time.
Artifact paths are resolved against the project root and only image files are
served. The UI does not replace or alter existing CLI behavior.

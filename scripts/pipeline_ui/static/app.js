const state = {
  overview: null,
  selectedDocument: null,
  selectedStage: "all",
  selectedArtifactPath: null,
  logOffset: 0,
  lastProcessSignature: "",
  artifactSearchTimer: null,
  aristotelLoaded: false,
  trainingLoaded: false,
  activeView: "artifact-trace",
  previewZoom: 1,
  previewPanX: 0,
  previewPanY: 0,
  previewDragging: false,
  previewDragX: 0,
  previewDragY: 0,
};

const elements = {
  activeCommand: document.querySelector("#active-command"),
  artifactPreview: document.querySelector("#artifact-preview"),
  artifactSearch: document.querySelector("#artifact-search"),
  artifactSummary: document.querySelector("#artifact-summary"),
  artifactWarning: document.querySelector("#artifact-warning"),
  aristotelSamples: document.querySelector("#aristotel-samples"),
  aristotelSummary: document.querySelector("#aristotel-summary"),
  clearConsole: document.querySelector("#clear-console"),
  consoleOutput: document.querySelector("#console-output"),
  contactSheet: document.querySelector("#contact-sheet"),
  documentCount: document.querySelector("#document-count"),
  documentList: document.querySelector("#document-list"),
  heartbeat: document.querySelector("#heartbeat"),
  previewCaption: document.querySelector("#preview-caption"),
  previewEmpty: document.querySelector("#preview-empty"),
  previewKind: document.querySelector("#preview-kind"),
  previewName: document.querySelector("#preview-name"),
  previewPath: document.querySelector("#preview-path"),
  previewViewport: document.querySelector("#preview-viewport"),
  projectRoot: document.querySelector("#project-root"),
  refreshArtifacts: document.querySelector("#refresh-artifacts"),
  refreshAristotel: document.querySelector("#refresh-aristotel"),
  stopButton: document.querySelector("#stop-button"),
  systemState: document.querySelector("#system-state"),
  toast: document.querySelector("#toast"),
  refreshTraining: document.querySelector("#refresh-training"),
  runMinosTraining: document.querySelector("#run-minos-training"),
  traceTitle: document.querySelector("#trace-title"),
  trainingRuns: document.querySelector("#training-runs"),
  trainingSummary: document.querySelector("#training-summary"),
  updatedAt: document.querySelector("#updated-at"),
  zoomControls: document.querySelector(".zoom-controls"),
  zoomIn: document.querySelector("#zoom-in"),
  zoomOut: document.querySelector("#zoom-out"),
  zoomReset: document.querySelector("#zoom-reset"),
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function showToast(message, isError = false) {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("visible");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(
    () => elements.toast.classList.remove("visible"),
    3200,
  );
}

function stageProgressMarkup(documentRecord) {
  return state.overview.stages
    .map((stage) => (
      `<span class="${documentRecord.stages[stage.id] ? "complete" : ""}" `
      + `title="${stage.label}: ${documentRecord.stages[stage.id] ? "complete" : "pending"}"></span>`
    ))
    .join("");
}

function renderDocuments() {
  const documents = state.overview.documents;
  elements.documentCount.textContent = documents.length;
  elements.documentList.innerHTML = "";

  if (!documents.length) {
    elements.documentList.innerHTML = (
      '<div class="empty-sheet">No documents found in handwritten_text.</div>'
    );
    state.selectedDocument = null;
    return;
  }

  if (!documents.some((item) => item.id === state.selectedDocument)) {
    state.selectedDocument = documents[0].id;
  }

  documents.forEach((record) => {
    const button = document.createElement("button");
    button.className = (
      `document-button ${record.id === state.selectedDocument ? "active" : ""}`
    );
    button.innerHTML = `
      <span class="document-name">${escapeHtml(record.id)}</span>
      <span class="document-progress">${stageProgressMarkup(record)}</span>
    `;
    button.addEventListener("click", () => {
      state.selectedDocument = record.id;
      state.selectedArtifactPath = null;
      renderDocuments();
      renderPhaseCompletion();
      loadArtifacts();
    });
    elements.documentList.appendChild(button);
  });
}

function renderPhaseCompletion() {
  const selected = state.overview?.documents.find(
    (item) => item.id === state.selectedDocument,
  );
  document.querySelectorAll(".phase-card").forEach((button) => {
    const stage = button.dataset.stage;
    button.classList.toggle("complete", Boolean(selected?.stages[stage]));
  });
  elements.traceTitle.textContent = selected
    ? `${selected.id} // artifact history`
    : "Select a document";
}

function processIsActive(process) {
  return ["running", "stopping"].includes(process?.status);
}

function renderProcess(process) {
  const active = processIsActive(process);
  const status = process?.status || "idle";
  const command = process?.command || "none";
  elements.activeCommand.textContent = command;
  elements.stopButton.disabled = !active;
  elements.systemState.textContent = active
    ? `${command.toUpperCase()} ${status.toUpperCase()}`
    : `SYSTEM ${status.toUpperCase()}`;
  elements.heartbeat.className = `heartbeat ${active ? "running" : status === "failed" ? "failed" : ""}`;

  document.querySelectorAll("[data-command]").forEach((button) => {
    if (button.id !== "stop-button") {
      button.disabled = active;
    }
    const isRunningCommand = active && button.dataset.command === command;
    button.classList.toggle("running", isRunningCommand);
  });
}

function setWorkspaceView(viewId) {
  state.activeView = viewId;
  document.querySelectorAll(".workspace-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewId);
  });
  document.querySelectorAll(".workspace-view").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.viewPanel === viewId);
  });

  if (viewId === "training-lab" && !state.trainingLoaded) {
    loadTrainingOverview();
  }
  if (viewId === "aristotel-lab" && !state.aristotelLoaded) {
    elements.aristotelSummary.textContent = (
      "Press Inspect 10 samples to generate a fresh teacher preview."
    );
  }
}

function renderOverview(payload) {
  state.overview = payload;
  elements.projectRoot.textContent = payload.base_dir;
  elements.updatedAt.textContent = `Telemetry ${new Date(payload.updated_at).toLocaleTimeString()}`;
  renderDocuments();
  renderPhaseCompletion();
  renderProcess(payload.process);
}

async function loadOverview({ refreshArtifacts = false } = {}) {
  try {
    const payload = await requestJson("/api/overview");
    const oldSignature = state.lastProcessSignature;
    const newSignature = JSON.stringify([
      payload.process.command,
      payload.process.status,
      payload.process.return_code,
    ]);
    state.lastProcessSignature = newSignature;
    renderOverview(payload);

    if (refreshArtifacts || (oldSignature && oldSignature !== newSignature
      && !processIsActive(payload.process))) {
      loadArtifacts();
    }
  } catch (error) {
    showToast(error.message, true);
    elements.systemState.textContent = "CONTROL ROOM OFFLINE";
    elements.heartbeat.className = "heartbeat failed";
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatMetricPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${Number(value).toFixed(2)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function applyPreviewTransform() {
  elements.artifactPreview.style.transform = (
    `translate(${state.previewPanX}px, ${state.previewPanY}px) `
    + `scale(${state.previewZoom})`
  );
  elements.zoomReset.textContent = `${Math.round(state.previewZoom * 100)}%`;
  elements.previewViewport.classList.toggle("zoomed", state.previewZoom > 1);
}

function resetPreviewZoom() {
  state.previewZoom = 1;
  state.previewPanX = 0;
  state.previewPanY = 0;
  applyPreviewTransform();
}

function setPreviewZoom(nextZoom) {
  state.previewZoom = Math.min(8, Math.max(0.5, nextZoom));
  if (state.previewZoom <= 1) {
    state.previewPanX = 0;
    state.previewPanY = 0;
  }
  applyPreviewTransform();
}

function selectArtifact(artifact, card) {
  state.selectedArtifactPath = artifact.relative_path;
  document.querySelectorAll(".artifact-card").forEach(
    (item) => item.classList.remove("active"),
  );
  card.classList.add("active");
  elements.previewEmpty.style.display = "none";
  elements.previewViewport.style.display = "flex";
  elements.zoomControls.classList.add("visible");
  resetPreviewZoom();
  elements.artifactPreview.src = artifact.url;
  elements.previewKind.textContent = `${artifact.stage} · ${artifact.kind}`;
  elements.previewName.textContent = artifact.name;
  elements.previewPath.textContent = artifact.relative_path;
  elements.previewPath.title = artifact.relative_path;
}

function renderArtifacts(payload) {
  const artifacts = payload.artifacts;
  elements.contactSheet.innerHTML = "";
  elements.artifactSummary.textContent = (
    `${payload.total_count} artifact${payload.total_count === 1 ? "" : "s"}`
  );
  elements.artifactWarning.textContent = payload.truncated
    ? `Showing first ${payload.returned_count}`
    : "";

  if (!artifacts.length) {
    elements.contactSheet.innerHTML = `
      <div class="empty-sheet">
        No images for this stage yet.<br>
        Run the phase, then refresh the trace.
      </div>
    `;
    return;
  }

  let selectedCard = null;
  let selectedArtifact = null;
  artifacts.forEach((artifact) => {
    const card = document.createElement("button");
    card.className = "artifact-card";
    card.innerHTML = `
      <span class="artifact-thumb">
        <img src="${artifact.url}" alt="" loading="lazy">
      </span>
      <span class="artifact-meta">
        <strong title="${escapeHtml(artifact.name)}">${escapeHtml(artifact.name)}</strong>
        <small>${escapeHtml(artifact.stage)} · ${escapeHtml(artifact.kind)} · ${formatBytes(artifact.size_bytes)}</small>
      </span>
    `;
    card.addEventListener("click", () => selectArtifact(artifact, card));
    elements.contactSheet.appendChild(card);

    if (artifact.relative_path === state.selectedArtifactPath) {
      selectedCard = card;
      selectedArtifact = artifact;
    }
  });

  if (selectedCard && selectedArtifact) {
    selectArtifact(selectedArtifact, selectedCard);
  } else if (artifacts.length) {
    selectArtifact(artifacts[0], elements.contactSheet.firstElementChild);
  }
}

async function loadArtifacts() {
  if (!state.selectedDocument) {
    return;
  }
  const params = new URLSearchParams({
    document: state.selectedDocument,
    stage: state.selectedStage,
    query: elements.artifactSearch.value.trim(),
    limit: "500",
  });
  elements.contactSheet.innerHTML = (
    '<div class="empty-sheet">Scanning pipeline artifacts...</div>'
  );
  try {
    renderArtifacts(await requestJson(`/api/artifacts?${params}`));
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderAristotelPreview(payload) {
  elements.aristotelSamples.innerHTML = "";
  if (payload.status !== "completed") {
    elements.aristotelSummary.textContent = (
      payload.message || "Aristotel preview is unavailable."
    );
    return;
  }

  elements.aristotelSummary.textContent = (
    `${payload.sample_count} samples · ${payload.recipe_count} active recipes · `
    + `output: ${payload.output_root}`
  );

  payload.samples.forEach((sample) => {
    const operations = (sample.operations || [])
      .map((operation) => {
        const name = operation.operation || operation.type || "operation";
        return (
          `<span title="${escapeHtml(JSON.stringify(operation))}">`
          + `${escapeHtml(name)}</span>`
        );
      })
      .join("");
    const cycle = (sample.defense_preview?.cycle || [])
      .map((step) => `<span>${escapeHtml(step)}</span>`)
      .join("");
    const card = document.createElement("article");
    card.className = "aristotel-card";
    card.innerHTML = `
      <div class="aristotel-card-top">
        <span class="aristotel-index">#${sample.sample_index}</span>
        <strong>${escapeHtml(sample.damage_recipe)}</strong>
        <span class="aristotel-label">${escapeHtml(sample.trust_label)}</span>
      </div>
      <div class="aristotel-image-pair">
        <figure>
          <img src="${sample.original_url}" alt="">
          <figcaption>original · ${escapeHtml(sample.label)}</figcaption>
        </figure>
        <figure>
          <img src="${sample.damaged_url}" alt="">
          <figcaption>damaged · ${formatPercent(sample.changed_pixel_ratio)}</figcaption>
        </figure>
      </div>
      <div class="aristotel-metrics">
        <span>changed ${sample.changed_pixel_count} px</span>
        <span>severity ${Number(sample.severity || 0).toFixed(2)}</span>
        <span title="${escapeHtml(sample.source_id || "")}">
          source ${escapeHtml(sample.source_id || "unknown")}
        </span>
      </div>
      <div class="aristotel-ops">${operations || "<span>no operation</span>"}</div>
      <div class="defense-preview">
        <strong>${escapeHtml(sample.defense_preview?.currently_available_tool || "diagnosis")}</strong>
        <p>${escapeHtml(sample.defense_preview?.note || "")}</p>
        <div class="defense-cycle">${cycle}</div>
      </div>
      <details class="aristotel-details">
        <summary>metadata</summary>
        <pre>${escapeHtml(JSON.stringify(sample.metadata, null, 2))}</pre>
      </details>
    `;
    elements.aristotelSamples.appendChild(card);
  });
}

async function loadAristotelPreview() {
  elements.aristotelSummary.textContent = "Asking Aristotel for 10 samples...";
  elements.aristotelSamples.innerHTML = "";
  elements.refreshAristotel.disabled = true;
  try {
    const payload = await requestJson("/api/aristotel-preview?limit=10");
    renderAristotelPreview(payload);
    state.aristotelLoaded = true;
  } catch (error) {
    showToast(error.message, true);
    elements.aristotelSummary.textContent = error.message;
  } finally {
    elements.refreshAristotel.disabled = false;
  }
}

function normalizeHistoryPoint(row, key) {
  const value = Number(row?.[key]);
  if (!Number.isFinite(value)) return null;
  if (key.includes("top")) return value * 100;
  return value;
}

function buildPolyline(history, key, width, height, padding, maxValue) {
  const points = history
    .map((row, index) => {
      const value = normalizeHistoryPoint(row, key);
      if (value === null) return null;
      const x = padding + (
        history.length <= 1
          ? 0
          : index * ((width - padding * 2) / (history.length - 1))
      );
      const y = height - padding - (
        Math.max(0, Math.min(value, maxValue)) / maxValue
      ) * (height - padding * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .filter(Boolean);
  return points.join(" ");
}

function renderTrainingChart(history) {
  if (!history?.length) {
    return '<div class="training-chart-empty">No epoch history saved yet.</div>';
  }

  const width = 520;
  const height = 170;
  const padding = 22;
  const keys = [
    ["val_top1", "Val Top-1", "orange"],
    ["val_top5", "Val Top-5", "green"],
    ["train_top1", "Train Top-1", "blue"],
  ];
  const polylines = keys
    .map(([key, label, color]) => {
      const points = buildPolyline(history, key, width, height, padding, 100);
      if (!points) return "";
      return `<polyline class="chart-line ${color}" points="${points}"><title>${label}</title></polyline>`;
    })
    .join("");
  const last = history[history.length - 1] || {};
  const legend = keys
    .map(([key, label, color]) => (
      `<span class="${color}">${label}: `
      + `${formatMetricPercent(normalizeHistoryPoint(last, key))}</span>`
    ))
    .join("");

  return `
    <div class="training-chart">
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Training accuracy history">
        <line class="chart-axis" x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}"></line>
        <line class="chart-axis" x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}"></line>
        <line class="chart-grid" x1="${padding}" y1="${padding}" x2="${width - padding}" y2="${padding}"></line>
        <line class="chart-grid" x1="${padding}" y1="${height / 2}" x2="${width - padding}" y2="${height / 2}"></line>
        ${polylines}
      </svg>
      <div class="training-chart-legend">${legend}</div>
    </div>
  `;
}

function renderRecipeScores(recipeScores) {
  if (!recipeScores?.length) {
    return '<div class="recipe-score-empty">No damage recipe metrics.</div>';
  }
  return recipeScores
    .map((score) => `
      <span title="count ${escapeHtml(score.count ?? "—")}">
        ${escapeHtml(score.name)} · ${formatMetricPercent(score.top5)}
      </span>
    `)
    .join("");
}

function renderTrainingOverview(payload) {
  if (payload.status !== "completed") {
    elements.trainingSummary.textContent = "Training telemetry unavailable.";
    return;
  }

  const process = payload.process || {};
  elements.trainingSummary.textContent = (
    `${payload.run_count} report folders · active process: `
    + `${process.command || "none"} / ${process.status || "idle"} · `
    + `reports: ${payload.report_root}`
  );
  elements.trainingRuns.innerHTML = "";

  if (!payload.runs.length) {
    elements.trainingRuns.innerHTML = (
      '<div class="training-empty">No training reports found yet.</div>'
    );
    return;
  }

  payload.runs.forEach((run) => {
    const split = run.split || {};
    const splitText = (
      split.train !== undefined
        ? `train ${split.train} · val ${split.validation} · test ${split.test}`
        : "split unavailable"
    );
    const card = document.createElement("article");
    card.className = "training-card";
    card.innerHTML = `
      <div class="training-card-head">
        <div>
          <strong>${escapeHtml(run.model_name)}</strong>
          <small>${escapeHtml(run.model_type || "unknown")} · ${escapeHtml(run.updated_at)}</small>
        </div>
        <span class="${run.model_exists ? "model-ok" : "model-missing"}">
          ${run.model_exists ? "model found" : "model missing"}
        </span>
      </div>
      <div class="training-metrics">
        <span><b>${formatMetricPercent(run.validation_top1)}</b><small>val top1</small></span>
        <span><b>${formatMetricPercent(run.validation_top5)}</b><small>val top5</small></span>
        <span><b>${formatMetricPercent(run.test_top1)}</b><small>test top1</small></span>
        <span><b>${formatMetricPercent(run.test_top5)}</b><small>test top5</small></span>
      </div>
      ${renderTrainingChart(run.training_history)}
      <div class="training-paths">
        <code>${escapeHtml(run.model_path || "no model path")}</code>
        <code>${escapeHtml(splitText)}</code>
        <code>${escapeHtml(run.dataset_jsonl || run.primary_report || "")}</code>
      </div>
      <div class="recipe-score-row">
        ${renderRecipeScores(run.recipe_scores)}
      </div>
    `;
    elements.trainingRuns.appendChild(card);
  });
}

async function loadTrainingOverview() {
  elements.trainingSummary.textContent = "Reading model reports...";
  try {
    const payload = await requestJson("/api/training-overview");
    renderTrainingOverview(payload);
    state.trainingLoaded = true;
  } catch (error) {
    showToast(error.message, true);
    elements.trainingSummary.textContent = error.message;
  }
}

function appendConsoleLines(lines, truncated = false) {
  if (truncated) {
    elements.consoleOutput.textContent += (
      "\n[control-room] Earlier output was truncated.\n"
    );
  }
  if (!lines.length) {
    return;
  }
  if (elements.consoleOutput.querySelector(".console-muted")) {
    elements.consoleOutput.textContent = "";
  }
  elements.consoleOutput.textContent += `${lines.join("\n")}\n`;
  elements.consoleOutput.scrollTop = elements.consoleOutput.scrollHeight;
}

async function pollLogs() {
  try {
    const payload = await requestJson(`/api/logs?offset=${state.logOffset}`);
    state.logOffset = payload.next_offset;
    appendConsoleLines(payload.lines, payload.truncated);
    renderProcess(payload.process);
  } catch (error) {
    console.error(error);
  }
}

async function runCommand(command) {
  if (command === "clean") {
    const confirmed = window.confirm(
      "Clean generated outputs? Inputs, models, datasets, and source code remain untouched.",
    );
    if (!confirmed) return;
  }

  state.logOffset = 0;
  elements.consoleOutput.textContent = "";
  try {
    const payload = await requestJson("/api/run", {
      method: "POST",
      body: JSON.stringify({ command }),
    });
    renderProcess(payload.process);
    showToast(`${command} launched`);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function stopCommand() {
  try {
    const payload = await requestJson("/api/stop", {
      method: "POST",
      body: "{}",
    });
    renderProcess(payload.process);
    showToast("Stop requested");
  } catch (error) {
    showToast(error.message, true);
  }
}

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", () => runCommand(button.dataset.command));
});

document.querySelectorAll(".workspace-tab").forEach((button) => {
  button.addEventListener("click", () => setWorkspaceView(button.dataset.view));
});

document.querySelectorAll(".stage-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".stage-tab").forEach(
      (item) => item.classList.remove("active"),
    );
    button.classList.add("active");
    state.selectedStage = button.dataset.stage;
    elements.artifactSearch.value = button.dataset.query || "";
    state.selectedArtifactPath = null;
    loadArtifacts();
  });
});

elements.stopButton.addEventListener("click", stopCommand);
elements.refreshArtifacts.addEventListener("click", () => {
  loadOverview({ refreshArtifacts: true });
});
elements.refreshAristotel.addEventListener("click", loadAristotelPreview);
elements.refreshTraining.addEventListener("click", loadTrainingOverview);
elements.runMinosTraining.addEventListener("click", () => runCommand("train"));
elements.clearConsole.addEventListener("click", () => {
  elements.consoleOutput.textContent = "";
});
elements.artifactSearch.addEventListener("input", () => {
  window.clearTimeout(state.artifactSearchTimer);
  state.artifactSearchTimer = window.setTimeout(loadArtifacts, 260);
});

elements.zoomIn.addEventListener("click", () => {
  setPreviewZoom(state.previewZoom * 1.25);
});
elements.zoomOut.addEventListener("click", () => {
  setPreviewZoom(state.previewZoom / 1.25);
});
elements.zoomReset.addEventListener("click", resetPreviewZoom);

elements.previewViewport.addEventListener("wheel", (event) => {
  event.preventDefault();
  setPreviewZoom(
    state.previewZoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15),
  );
}, { passive: false });

elements.previewViewport.addEventListener("dblclick", resetPreviewZoom);

elements.previewViewport.addEventListener("pointerdown", (event) => {
  if (state.previewZoom <= 1) return;
  state.previewDragging = true;
  state.previewDragX = event.clientX - state.previewPanX;
  state.previewDragY = event.clientY - state.previewPanY;
  elements.previewViewport.classList.add("dragging");
  elements.previewViewport.setPointerCapture(event.pointerId);
});

elements.previewViewport.addEventListener("pointermove", (event) => {
  if (!state.previewDragging) return;
  state.previewPanX = event.clientX - state.previewDragX;
  state.previewPanY = event.clientY - state.previewDragY;
  applyPreviewTransform();
});

function stopPreviewDrag(event) {
  state.previewDragging = false;
  elements.previewViewport.classList.remove("dragging");
  if (event.pointerId !== undefined
      && elements.previewViewport.hasPointerCapture(event.pointerId)) {
    elements.previewViewport.releasePointerCapture(event.pointerId);
  }
}

elements.previewViewport.addEventListener("pointerup", stopPreviewDrag);
elements.previewViewport.addEventListener("pointercancel", stopPreviewDrag);

async function boot() {
  const urlParameters = new URLSearchParams(window.location.search);
  const requestedStage = urlParameters.get("stage");
  const requestedQuery = urlParameters.get("query");
  if (requestedStage) {
    state.selectedStage = requestedStage;
  }
  if (requestedQuery) {
    elements.artifactSearch.value = requestedQuery;
  }
  document.querySelectorAll(".stage-tab").forEach((button) => {
    const matchesStage = button.dataset.stage === state.selectedStage;
    const buttonQuery = button.dataset.query || "";
    const matchesQuery = buttonQuery === elements.artifactSearch.value;
    button.classList.toggle("active", matchesStage && matchesQuery);
  });
  await loadOverview({ refreshArtifacts: true });
  await pollLogs();
  window.setInterval(pollLogs, 700);
  window.setInterval(loadOverview, 2200);
}

boot();

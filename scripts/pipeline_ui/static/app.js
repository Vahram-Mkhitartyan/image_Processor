const state = {
  overview: null,
  selectedDocument: null,
  selectedStage: "all",
  selectedArtifactPath: null,
  logOffset: 0,
  lastProcessSignature: "",
  artifactSearchTimer: null,
};

const elements = {
  activeCommand: document.querySelector("#active-command"),
  artifactPreview: document.querySelector("#artifact-preview"),
  artifactSearch: document.querySelector("#artifact-search"),
  artifactSummary: document.querySelector("#artifact-summary"),
  artifactWarning: document.querySelector("#artifact-warning"),
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
  projectRoot: document.querySelector("#project-root"),
  refreshArtifacts: document.querySelector("#refresh-artifacts"),
  stopButton: document.querySelector("#stop-button"),
  systemState: document.querySelector("#system-state"),
  toast: document.querySelector("#toast"),
  traceTitle: document.querySelector("#trace-title"),
  updatedAt: document.querySelector("#updated-at"),
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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function selectArtifact(artifact, card) {
  state.selectedArtifactPath = artifact.relative_path;
  document.querySelectorAll(".artifact-card").forEach(
    (item) => item.classList.remove("active"),
  );
  card.classList.add("active");
  elements.previewEmpty.style.display = "none";
  elements.artifactPreview.style.display = "block";
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
    limit: "240",
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

document.querySelectorAll(".stage-tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".stage-tab").forEach(
      (item) => item.classList.remove("active"),
    );
    button.classList.add("active");
    state.selectedStage = button.dataset.stage;
    state.selectedArtifactPath = null;
    loadArtifacts();
  });
});

elements.stopButton.addEventListener("click", stopCommand);
elements.refreshArtifacts.addEventListener("click", () => {
  loadOverview({ refreshArtifacts: true });
});
elements.clearConsole.addEventListener("click", () => {
  elements.consoleOutput.textContent = "";
});
elements.artifactSearch.addEventListener("input", () => {
  window.clearTimeout(state.artifactSearchTimer);
  state.artifactSearchTimer = window.setTimeout(loadArtifacts, 260);
});

async function boot() {
  await loadOverview({ refreshArtifacts: true });
  await pollLogs();
  window.setInterval(pollLogs, 700);
  window.setInterval(loadOverview, 2200);
}

boot();

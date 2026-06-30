const state = {
  overview: null,
  selectedDocument: null,
  selectedStage: "all",
  selectedArtifactPath: null,
  artifactContextRequestId: 0,
  cropContext: null,
  logOffset: 0,
  lastProcessSignature: "",
  artifactSearchTimer: null,
  aristotelLoaded: false,
  aristotelSamples: [],
  trainingLoaded: false,
  scrilogLoaded: false,
  scrilogWorkspace: null,
  scrilogIndex: 0,
  scrilogClassFilter: "",
  scrilogDirty: false,
  scrististicsLoaded: false,
  scrististicsClass: "",
  scrististicsFeature: "endpoints",
  activeView: "artifact-trace",
  previewZoom: 1,
  previewPanX: 0,
  previewPanY: 0,
  previewDragging: false,
  previewDragX: 0,
  previewDragY: 0,
  detailZoom: 1,
  detailPanX: 0,
  detailPanY: 0,
  detailDragging: false,
  detailDragX: 0,
  detailDragY: 0,
  decisionCanvas: null,
  printedContextCanvas: null,
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
  cropContextBox: document.querySelector("#crop-context-box"),
  cropContextFrame: document.querySelector("#crop-context-frame"),
  cropContextImage: document.querySelector("#crop-context-image"),
  cropContextMeta: document.querySelector("#crop-context-meta"),
  cropContextPanel: document.querySelector("#crop-context-panel"),
  cropContextTitle: document.querySelector("#crop-context-title"),
  decisionCanvasEmpty: document.querySelector("#decision-canvas-empty"),
  decisionCanvasFrame: document.querySelector("#decision-canvas-frame"),
  decisionCanvasImage: document.querySelector("#decision-canvas-image"),
  decisionCanvasOverlay: document.querySelector("#decision-canvas-overlay"),
  decisionCanvasSummary: document.querySelector("#decision-canvas-summary"),
  documentCount: document.querySelector("#document-count"),
  documentList: document.querySelector("#document-list"),
  elevatorDown: document.querySelector("#elevator-down"),
  elevatorUp: document.querySelector("#elevator-up"),
  heartbeat: document.querySelector("#heartbeat"),
  hardRefreshUi: document.querySelector("#hard-refresh-ui"),
  previewCaption: document.querySelector("#preview-caption"),
  previewEmpty: document.querySelector("#preview-empty"),
  previewKind: document.querySelector("#preview-kind"),
  previewName: document.querySelector("#preview-name"),
  previewPath: document.querySelector("#preview-path"),
  previewViewport: document.querySelector("#preview-viewport"),
  printedContextEmpty: document.querySelector("#printed-context-empty"),
  printedContextFrame: document.querySelector("#printed-context-frame"),
  printedContextImage: document.querySelector("#printed-context-image"),
  printedContextOverlay: document.querySelector("#printed-context-overlay"),
  printedContextSummary: document.querySelector("#printed-context-summary"),
  projectRoot: document.querySelector("#project-root"),
  refreshArtifacts: document.querySelector("#refresh-artifacts"),
  refreshDecisionCanvas: document.querySelector("#refresh-decision-canvas"),
  refreshPrintedContext: document.querySelector("#refresh-printed-context"),
  refreshAristotel: document.querySelector("#refresh-aristotel"),
  sampleDetail: document.querySelector("#sample-detail"),
  sampleDetailBackdrop: document.querySelector("#sample-detail-backdrop"),
  sampleDetailClose: document.querySelector("#sample-detail-close"),
  sampleDetailContent: document.querySelector("#sample-detail-content"),
  sidebarPanels: document.querySelectorAll(".sidebar-view"),
  stopButton: document.querySelector("#stop-button"),
  systemState: document.querySelector("#system-state"),
  toast: document.querySelector("#toast"),
  refreshTraining: document.querySelector("#refresh-training"),
  runMinosTraining: document.querySelector("#run-minos-training"),
  scrilogClassBadge: document.querySelector("#scrilog-class-badge"),
  scrilogClassFilter: document.querySelector("#scrilog-class-filter"),
  scrilogExport: document.querySelector("#scrilog-export"),
  scrilogFields: document.querySelector("#scrilog-fields"),
  scrilogImage: document.querySelector("#scrilog-image"),
  scrilogImageName: document.querySelector("#scrilog-image-name"),
  scrilogIndex: document.querySelector("#scrilog-index"),
  scrilogNext: document.querySelector("#scrilog-next"),
  scrilogNotes: document.querySelector("#scrilog-notes"),
  scrilogOutputPath: document.querySelector("#scrilog-output-path"),
  scrilogPrevious: document.querySelector("#scrilog-previous"),
  scrilogReset: document.querySelector("#scrilog-reset"),
  scrilogSave: document.querySelector("#scrilog-save"),
  scrilogSaveNext: document.querySelector("#scrilog-save-next"),
  scrilogSaveState: document.querySelector("#scrilog-save-state"),
  scrilogSavedCount: document.querySelector("#scrilog-saved-count"),
  scrilogSidebarExport: document.querySelector("#scrilog-sidebar-export"),
  scrilogSidebarSave: document.querySelector("#scrilog-sidebar-save"),
  scrilogSourceId: document.querySelector("#scrilog-source-id"),
  scrilogStatus: document.querySelector("#scrilog-status"),
  scrilogTotal: document.querySelector("#scrilog-total"),
  scrististicsChart: document.querySelector("#scrististics-chart"),
  scrististicsChartTitle: document.querySelector("#scrististics-chart-title"),
  scrististicsClass: document.querySelector("#scrististics-class"),
  scrististicsClassBadge: document.querySelector("#scrististics-class-badge"),
  scrististicsDatasetSummary: document.querySelector("#scrististics-dataset-summary"),
  scrististicsFeature: document.querySelector("#scrististics-feature"),
  scrististicsKeyMetrics: document.querySelector("#scrististics-key-metrics"),
  scrististicsMode: document.querySelector("#scrististics-mode"),
  scrististicsProfilePath: document.querySelector("#scrististics-profile-path"),
  scrististicsRefresh: document.querySelector("#scrististics-refresh"),
  scrististicsVariants: document.querySelector("#scrististics-variants"),
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

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "readonly");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();

  try {
    document.execCommand("copy");
  } finally {
    textarea.remove();
  }
}

function hardRefreshUi() {
  const url = new URL(window.location.href);
  url.searchParams.set("ui_refresh", Date.now().toString());
  window.location.replace(url.toString());
}

function activeElevatorTarget() {
  if (elements.sampleDetail?.classList.contains("visible")) {
    const detailCard = elements.sampleDetail.querySelector(".sample-detail-card");
    if (detailCard && detailCard.scrollHeight > detailCard.clientHeight) {
      return detailCard;
    }
  }
  return document.scrollingElement || document.documentElement;
}

function scrollContainerTo(container, targetTop) {
  const startTop = container.scrollTop;
  const distance = targetTop - startTop;
  if (Math.abs(distance) < 4) return;

  const duration = Math.min(720, Math.max(260, Math.abs(distance) * 0.16));
  const startedAt = performance.now();
  const easeOutCubic = (value) => 1 - Math.pow(1 - value, 3);

  function step(now) {
    const progress = Math.min(1, (now - startedAt) / duration);
    container.scrollTop = startTop + distance * easeOutCubic(progress);
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  }

  window.requestAnimationFrame(step);
}

function elevatorScroll(direction) {
  const target = activeElevatorTarget();
  const maxTop = Math.max(0, target.scrollHeight - target.clientHeight);
  scrollContainerTo(target, direction === "up" ? 0 : maxTop);
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
      state.decisionCanvas = null;
      renderDocuments();
      renderPhaseCompletion();
      loadArtifacts();
      loadDecisionCanvas();
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
  elements.sidebarPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.sidebarPanel === viewId);
  });

  if (viewId === "training-lab" && !state.trainingLoaded) {
    loadTrainingOverview();
  }
  if (viewId === "aristotel-lab" && !state.aristotelLoaded) {
    elements.aristotelSummary.textContent = (
      "Press Inspect 10 samples to generate a fresh teacher preview."
    );
  }
  if (viewId === "scrilog-lab" && !state.scrilogLoaded) {
    loadScrilogWorkspace(0);
  }
  if (viewId === "scrilog-lab" && !state.scrististicsLoaded) {
    loadScrististicsDistribution();
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

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(3);
}

function shortStageName(stage) {
  return String(stage || "unknown")
    .replace("stage_", "")
    .replaceAll("_", " ");
}

function renderStageTimeline(reconstruction) {
  const records = reconstruction.stage_records || [];
  const plan = reconstruction.stage_defense_plan || {};
  const stages = records.length
    ? records
    : Object.entries(plan).map(([stage, names], index) => ({
      stage,
      stage_index: index,
      routed_defense_types: names || [],
      implemented_defense_types: [],
      generated_hypothesis_ids: [],
      generated_count: 0,
      notes: [],
    }));

  if (!stages.length) {
    return '<div class="stage-timeline empty">No staged defense plan emitted.</div>';
  }

  return `
    <div class="stage-timeline">
      ${stages.map((stage) => {
        const routed = stage.routed_defense_types || [];
        const implemented = stage.implemented_defense_types || [];
        const generated = Number(stage.generated_count || 0);
        const state = generated > 0
          ? "generated"
          : routed.length
            ? "routed"
            : "empty";
        const notes = (stage.notes || []).slice(0, 2)
          .map((note) => `<span>${escapeHtml(note)}</span>`)
          .join("");
        return `
          <article class="stage-timeline-card ${state}">
            <strong>${escapeHtml(shortStageName(stage.stage))}</strong>
            <small>r${routed.length} · i${implemented.length} · h${generated}</small>
            <div>
              ${
                routed.length
                  ? routed.map((name) => `<span>${escapeHtml(name)}</span>`).join("")
                  : "<span>no routed tool</span>"
              }
            </div>
            ${notes ? `<p>${notes}</p>` : ""}
          </article>
        `;
      }).join("")}
    </div>
  `;
}

function renderDefenseToolSummary(reconstruction) {
  const tools = reconstruction.tool_summary || [];
  if (!tools.length) {
    return '<div class="defense-tool-grid empty">No routed defenses.</div>';
  }
  const cards = tools
    .map((tool) => `
      <span
        class="defense-tool-chip ${escapeHtml(tool.state || "unknown")}"
        title="${escapeHtml(JSON.stringify(tool))}"
      >
        <strong>${escapeHtml(tool.defense_name)}</strong>
        <small>
          ${escapeHtml(shortStageName(tool.stage))}
          · ${escapeHtml(tool.state || "unknown")}
          · c${escapeHtml(tool.candidate_count ?? 0)}
          · a${escapeHtml(tool.accepted_count ?? 0)}
        </small>
      </span>
    `)
    .join("");
  return `<div class="defense-tool-grid">${cards}</div>`;
}

function renderHypothesisMiniCards(reconstruction) {
  const hypotheses = reconstruction.hypotheses || [];
  if (!hypotheses.length) {
    return '<div class="hypothesis-empty">No repair hypotheses generated.</div>';
  }
  return `
    <div class="hypothesis-strip">
      ${hypotheses.map((hypothesis) => {
        const firstImage = hypothesis.primary_image_url
          || hypothesis.image_steps?.[0]?.url
          || "";
        const status = hypothesis.selected
          ? "selected"
          : hypothesis.accepted
            ? "accepted"
            : "rejected";
        const reasons = (hypothesis.rejection_reasons || []).slice(0, 2)
          .map((reason) => `<span>${escapeHtml(reason)}</span>`)
          .join("");
        const chain = (hypothesis.defense_chain || [])
          .map((name) => escapeHtml(name))
          .join(" → ");
        const parent = hypothesis.debug_reference_hypothesis_id
          || hypothesis.branch_parent_hypothesis_id
          || "h0_original";
        return `
          <article class="hypothesis-mini ${status}">
            <div class="hypothesis-mini-image">
              ${
                firstImage
                  ? `<img src="${firstImage}" alt="">`
                  : '<span>no image</span>'
              }
            </div>
            <div class="hypothesis-mini-body">
              <strong>${escapeHtml(hypothesis.hypothesis_id || "hypothesis")}</strong>
              <span>${escapeHtml(hypothesis.defense_name || "unknown")}</span>
              <small>${escapeHtml(shortStageName(hypothesis.stage))} · ${status} · score ${formatScore(hypothesis.score)}</small>
              <small title="${chain}">from ${escapeHtml(parent)}</small>
              <div class="hypothesis-reasons">${reasons || "<span>passes verifier</span>"}</div>
            </div>
          </article>
        `;
      }).join("")}
    </div>
  `;
}

function renderLineRemovalSequences(reconstruction, options = {}) {
  const sequences = reconstruction.line_removal_sequences || [];
  if (!sequences.length) {
    return "";
  }

  const compact = Boolean(options.compact);
  return `
    <section class="line-removal-sequences ${compact ? "compact" : ""}">
      <div class="line-removal-title">
        <strong>Line-removal sequence</strong>
        <span>${sequences.length} candidate${sequences.length === 1 ? "" : "s"}</span>
      </div>
      ${sequences.map((sequence) => {
        const cleanup = sequence.cleanup || {};
        const bridge = sequence.bridge || null;
        const images = sequence.images || [];
        const tags = [
          cleanup.hypothesis_id ? `cut ${cleanup.hypothesis_id}` : null,
          bridge?.hypothesis_id ? `bridge ${bridge.hypothesis_id}` : null,
          cleanup.accepted ? "cut accepted" : "cut rejected",
          bridge ? (bridge.accepted ? "bridge accepted" : "bridge rejected") : "no bridge",
        ].filter(Boolean);

        return `
          <article class="line-removal-sequence">
            <header>
              <div>
                <strong>${escapeHtml(cleanup.defense_name || "linear_artifact_removal")}</strong>
                <small>${escapeHtml(cleanup.hypothesis_id || "unknown")}</small>
              </div>
              <span>${formatScore(cleanup.score)}</span>
            </header>
            <div class="line-removal-images">
              ${images.map((image) => `
                <figure>
                  <img src="${image.url}" alt="">
                  <figcaption>${escapeHtml(image.label || image.step || "image")}</figcaption>
                </figure>
              `).join("")}
            </div>
            <div class="line-removal-tags">
              ${tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}
            </div>
          </article>
        `;
      }).join("")}
    </section>
  `;
}

function renderHypothesisWorkbench(reconstruction) {
  const hypotheses = reconstruction.hypotheses || [];
  if (!hypotheses.length) {
    return '<div class="sample-detail-defense muted"><strong>hypotheses</strong><div><span>none generated</span></div></div>';
  }

  return `
    <section class="hypothesis-workbench">
      <div class="hypothesis-workbench-title">
        <strong>Hypothesis workbench</strong>
        <span>${hypotheses.length} candidate${hypotheses.length === 1 ? "" : "s"}</span>
      </div>
      <div class="hypothesis-workbench-grid">
        ${hypotheses.map((hypothesis) => {
          const status = hypothesis.selected
            ? "selected"
            : hypothesis.accepted
              ? "accepted"
              : "rejected";
          const imageSteps = (hypothesis.image_steps || [])
            .map((image) => `
              <figure>
                <img src="${image.url}" alt="">
                <figcaption>${escapeHtml(image.step)} · ${escapeHtml(image.label)}</figcaption>
              </figure>
            `)
            .join("");
          const reasons = (hypothesis.rejection_reasons || [])
            .map((reason) => `<span>${escapeHtml(reason)}</span>`)
            .join("");
          const deltas = Object.entries(hypothesis.topology_delta || {})
            .map(([key, value]) => (
              `<span><b>${escapeHtml(key.replaceAll("_", " "))}</b> `
              + `${escapeHtml(value)}</span>`
            ))
            .join("");
          const chain = hypothesis.defense_chain || [];
          const chainTags = chain
            .map((name, index) => `<span>${index + 1}. ${escapeHtml(name)}</span>`)
            .join("");
          const parent = hypothesis.debug_reference_hypothesis_id
            || hypothesis.branch_parent_hypothesis_id
            || "h0_original";
          const phaseTags = [
            `compare: ${hypothesis.debug_reference === "parent_branch" ? "previous phase" : "original"}`,
            `source: ${parent}`,
            hypothesis.phase_changed_ink_ratio !== null && hypothesis.phase_changed_ink_ratio !== undefined
              ? `phase changed: ${formatPercent(hypothesis.phase_changed_ink_ratio)}`
              : null,
            hypothesis.phase_added_ink_pixels !== null && hypothesis.phase_added_ink_pixels !== undefined
              ? `added: ${hypothesis.phase_added_ink_pixels}px`
              : null,
            hypothesis.phase_removed_ink_pixels !== null && hypothesis.phase_removed_ink_pixels !== undefined
              ? `removed: ${hypothesis.phase_removed_ink_pixels}px`
              : null,
          ]
            .filter(Boolean)
            .map((item) => `<span>${escapeHtml(item)}</span>`)
            .join("");
          return `
            <article class="hypothesis-detail ${status}">
              <header>
                <div>
                  <strong>${escapeHtml(hypothesis.hypothesis_id || "hypothesis")}</strong>
                  <small>${escapeHtml(hypothesis.defense_name || "unknown defense")}</small>
                </div>
                <span>${escapeHtml(shortStageName(hypothesis.stage))} · ${status} · ${formatScore(hypothesis.score)}</span>
              </header>
              <div class="hypothesis-detail-images">
                ${imageSteps || '<div class="process-empty">No images for this hypothesis.</div>'}
              </div>
              <div class="hypothesis-detail-row phase">
                <strong>comparison</strong>
                <div class="hypothesis-detail-tags">${phaseTags}</div>
              </div>
              <div class="hypothesis-detail-row chain">
                <strong>defense chain</strong>
                <div class="hypothesis-detail-tags">${chainTags || "<span>no defense chain</span>"}</div>
              </div>
              <div class="hypothesis-detail-row">
                <strong>verifier</strong>
                <div class="hypothesis-detail-tags">${reasons || "<span>passes verifier</span>"}</div>
              </div>
              <div class="hypothesis-detail-row muted">
                <strong>topology delta</strong>
                <div class="hypothesis-detail-tags">${deltas || "<span>no topology delta</span>"}</div>
              </div>
            </article>
          `;
        }).join("")}
      </div>
    </section>
  `;
}

function sampleResultJsonPath(sample) {
  const reconstruction = sample.reconstruction_preview || {};
  return (
    reconstruction.result_json_path
    || reconstruction.result_json_url
    || sample.result_json_path
    || sample.result_json_url
    || ""
  );
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

function hideCropContext(reason = "") {
  state.cropContext = null;
  elements.cropContextPanel.classList.remove("visible");
  elements.previewViewport.classList.remove("with-context");
  elements.cropContextImage.removeAttribute("src");
  elements.cropContextBox.style.display = "none";
  elements.cropContextMeta.textContent = reason || "No crop context available.";
}

function renderCropContextBox() {
  const context = state.cropContext;
  const image = elements.cropContextImage;
  if (!context?.available || !image.naturalWidth || !image.naturalHeight) {
    elements.cropContextBox.style.display = "none";
    return;
  }

  const bbox = context.bbox;
  const imageRect = image.getBoundingClientRect();
  const frameRect = elements.cropContextFrame.getBoundingClientRect();
  const scaleX = imageRect.width / image.naturalWidth;
  const scaleY = imageRect.height / image.naturalHeight;
  elements.cropContextBox.style.display = "block";
  elements.cropContextBox.style.left = `${imageRect.left - frameRect.left + bbox.x1 * scaleX}px`;
  elements.cropContextBox.style.top = `${imageRect.top - frameRect.top + bbox.y1 * scaleY}px`;
  elements.cropContextBox.style.width = `${Math.max(2, (bbox.x2 - bbox.x1) * scaleX)}px`;
  elements.cropContextBox.style.height = `${Math.max(2, (bbox.y2 - bbox.y1) * scaleY)}px`;
}

function renderCropContext(context) {
  if (!context?.available) {
    hideCropContext(context?.reason ? `Context: ${context.reason}` : "");
    return;
  }

  state.cropContext = context;
  const record = context.record || {};
  const bbox = context.bbox || {};
  elements.cropContextPanel.classList.add("visible");
  elements.previewViewport.classList.add("with-context");
  elements.cropContextTitle.textContent = (
    `${record.layer || "crop"} · ${record.source_group_id || "source group"}`
  );
  elements.cropContextMeta.innerHTML = `
    <span>TU ${escapeHtml(record.text_unit_id ?? "—")}</span>
    <span>${escapeHtml(record.recommended_next_node || "route unknown")}</span>
    <code>x${bbox.x1}, y${bbox.y1}, w${bbox.x2 - bbox.x1}, h${bbox.y2 - bbox.y1}</code>
    ${
      record.parent_stacked_source_group_id
        ? `<span class="context-warning">split from ${escapeHtml(record.parent_stacked_source_group_id)}</span>`
        : ""
    }
  `;
  elements.cropContextBox.style.display = "none";
  elements.cropContextImage.src = context.source_document_url;
}

async function loadArtifactContext(artifact) {
  const requestId = ++state.artifactContextRequestId;
  const isN02CropArtifact = (
    artifact
    && artifact.stage === "n02"
    && artifact.relative_path.includes("/n02_crop_refiner/crops/")
  );
  if (!isN02CropArtifact) {
    hideCropContext("");
    return;
  }

  try {
    const params = new URLSearchParams({
      document: state.selectedDocument,
      path: artifact.relative_path,
    });
    const context = await requestJson(`/api/artifact-context?${params}`);
    if (requestId !== state.artifactContextRequestId) {
      return;
    }
    renderCropContext(context);
  } catch (error) {
    if (requestId === state.artifactContextRequestId) {
      hideCropContext(`Context failed: ${error.message}`);
    }
  }
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
  loadArtifactContext(artifact);
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
    if (state.selectedStage === "n04") {
      loadPrintedContextCanvas();
    }
    if (state.selectedStage === "n05") {
      loadDecisionCanvas();
    }
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderPrintedContextBoxes() {
  const payload = state.printedContextCanvas;
  const image = elements.printedContextImage;
  const overlay = elements.printedContextOverlay;
  overlay.innerHTML = "";

  if (!payload?.available || !image.naturalWidth || !image.naturalHeight) {
    return;
  }

  const imageRect = image.getBoundingClientRect();
  const frameRect = elements.printedContextFrame.getBoundingClientRect();
  const scaleX = imageRect.width / image.naturalWidth;
  const scaleY = imageRect.height / image.naturalHeight;

  (payload.boxes || []).forEach((box) => {
    const bbox = box.bbox || {};
    const left = imageRect.left - frameRect.left + bbox.x1 * scaleX;
    const top = imageRect.top - frameRect.top + bbox.y1 * scaleY;
    const width = Math.max(14, (bbox.x2 - bbox.x1) * scaleX);
    const height = Math.max(12, (bbox.y2 - bbox.y1) * scaleY);
    const candidates = (box.candidates || [])
      .map((candidate, index) => `${index + 1}. ${candidate.text}`)
      .join(" · ");
    const marker = document.createElement("button");
    marker.className = [
      "decision-text-box",
      "printed-context-box",
      box.analysis_mask_used_for_tesseract ? "mask-source" : "",
      box.text ? "ready" : "weak",
    ].filter(Boolean).join(" ");
    marker.style.left = `${left}px`;
    marker.style.top = `${top}px`;
    marker.style.width = `${width}px`;
    marker.style.height = `${height}px`;
    marker.title = [
      `token: ${box.token_id || "unknown"}`,
      `layer: ${box.layer || "unknown"}`,
      `visual: ${box.visual_class || "unknown"}`,
      `source: ${box.tesseract_input_source || "unknown"}`,
      `bbox source: ${box.n04_crop_bbox_source || "route_bbox"}`,
      `derived: ${box.black_mask_derived_bbox_reason || "none"}`,
      `ocr mask: ${box.black_mask_reflected_ocr_source_kind || "regular_crop"}`,
      `mask: ${box.mask_source || "none"}`,
      `candidates: ${candidates || "none"}`,
    ].join("\n");
    marker.innerHTML = `
      <span class="decision-text">${escapeHtml(box.text || "∅")}</span>
      <span class="decision-score">${escapeHtml(String(box.layer || "raw"))}</span>
    `;
    marker.addEventListener("click", () => {
      const copied = candidates || box.text || "";
      copyTextToClipboard(copied);
      showToast(`Copied N04 candidates for ${box.token_id || "printed token"}`);
    });
    overlay.appendChild(marker);
  });
}

function renderPrintedContextCanvas(payload) {
  state.printedContextCanvas = payload;
  if (!payload?.available) {
    elements.printedContextFrame.classList.remove("visible");
    elements.printedContextEmpty.style.display = "grid";
    elements.printedContextEmpty.textContent = (
      payload?.reason
        ? `Printed context unavailable: ${payload.reason}`
        : "Run N04, then refresh this canvas to see printed OCR context tokens."
    );
    elements.printedContextSummary.textContent = "No canvas";
    return;
  }

  const summary = payload.summary || {};
  elements.printedContextSummary.textContent = (
    `${summary.box_count || 0} boxes · `
    + `${summary.boxes_with_text || 0} with text · `
    + `${summary.black_mask_box_count || 0} black-mask · `
    + `${payload.source_document_kind || "canvas"}`
  );
  elements.printedContextEmpty.style.display = "none";
  elements.printedContextFrame.classList.add("visible");
  elements.printedContextOverlay.innerHTML = "";
  elements.printedContextImage.src = payload.source_document_url;
  if (elements.printedContextImage.complete) {
    renderPrintedContextBoxes();
  }
}

async function loadPrintedContextCanvas() {
  if (!state.selectedDocument) {
    return;
  }
  elements.printedContextSummary.textContent = "Reading N04 printed context...";
  try {
    const params = new URLSearchParams({ document: state.selectedDocument });
    renderPrintedContextCanvas(
      await requestJson(`/api/n04-printed-context-canvas?${params}`),
    );
  } catch (error) {
    elements.printedContextSummary.textContent = "Canvas failed";
    showToast(error.message, true);
  }
}

function renderDecisionCanvasBoxes() {
  const payload = state.decisionCanvas;
  const image = elements.decisionCanvasImage;
  const overlay = elements.decisionCanvasOverlay;
  overlay.innerHTML = "";

  if (!payload?.available || !image.naturalWidth || !image.naturalHeight) {
    return;
  }

  const imageRect = image.getBoundingClientRect();
  const frameRect = elements.decisionCanvasFrame.getBoundingClientRect();
  const scaleX = imageRect.width / image.naturalWidth;
  const scaleY = imageRect.height / image.naturalHeight;

  (payload.boxes || []).forEach((box) => {
    const bbox = box.bbox || {};
    const left = imageRect.left - frameRect.left + bbox.x1 * scaleX;
    const top = imageRect.top - frameRect.top + bbox.y1 * scaleY;
    const width = Math.max(14, (bbox.x2 - bbox.x1) * scaleX);
    const height = Math.max(12, (bbox.y2 - bbox.y1) * scaleY);
    const marker = document.createElement("button");
    marker.className = `decision-text-box ${box.status === "provisional_ready" ? "ready" : "weak"}`;
    marker.style.left = `${left}px`;
    marker.style.top = `${top}px`;
    marker.style.width = `${width}px`;
    marker.style.height = `${height}px`;
    marker.title = [
      `unit: ${box.text_unit_id || "unknown"}`,
      `status: ${box.status || "unknown"}`,
      `score: ${box.score ?? "—"}`,
      `backup: ${box.backup_string || "none"}`,
    ].join("\n");
    marker.innerHTML = `
      <span class="decision-text">${escapeHtml(box.selected_text || "∅")}</span>
      <span class="decision-score">${escapeHtml(String(box.status || "unknown"))}</span>
    `;
    marker.addEventListener("click", () => {
      copyTextToClipboard(box.backup_string || box.selected_text || "");
      showToast(`Copied ${box.backup_string ? "backup string" : "text"} for ${box.text_unit_id || "unit"}`);
    });
    overlay.appendChild(marker);
  });
}

function renderDecisionCanvas(payload) {
  state.decisionCanvas = payload;
  if (!payload?.available) {
    elements.decisionCanvasFrame.classList.remove("visible");
    elements.decisionCanvasEmpty.style.display = "grid";
    elements.decisionCanvasEmpty.textContent = (
      payload?.reason
        ? `Decision canvas unavailable: ${payload.reason}`
        : "Run N05, then refresh this canvas to see provisional OCR text placed by bbox."
    );
    elements.decisionCanvasSummary.textContent = "No canvas";
    return;
  }

  const summary = payload.summary || {};
  elements.decisionCanvasSummary.textContent = (
    `${summary.box_count || 0} boxes · ${summary.ready_count || 0} ready`
  );
  elements.decisionCanvasEmpty.style.display = "none";
  elements.decisionCanvasFrame.classList.add("visible");
  elements.decisionCanvasOverlay.innerHTML = "";
  elements.decisionCanvasImage.src = payload.source_document_url;
  if (elements.decisionCanvasImage.complete) {
    renderDecisionCanvasBoxes();
  }
}

async function loadDecisionCanvas() {
  if (!state.selectedDocument) {
    return;
  }
  elements.decisionCanvasSummary.textContent = "Reading N05 decision matrix...";
  try {
    const params = new URLSearchParams({ document: state.selectedDocument });
    renderDecisionCanvas(await requestJson(`/api/n05-decision-canvas?${params}`));
  } catch (error) {
    elements.decisionCanvasSummary.textContent = "Canvas failed";
    showToast(error.message, true);
  }
}

function renderAristotelPreview(payload) {
  elements.aristotelSamples.innerHTML = "";
  state.aristotelSamples = payload.samples || [];
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
    const reconstruction = sample.reconstruction_preview || {};
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
    const allowedDefenses = (reconstruction.allowed_defense_types || [])
      .map((name) => `<span>${escapeHtml(name)}</span>`)
      .join("");
    const unsupportedDefenses = (reconstruction.unsupported_defense_types || [])
      .map((name) => `<span>${escapeHtml(name)}</span>`)
      .join("");
    const damageReasons = (reconstruction.damage_reasons || [])
      .map((name) => `<span>${escapeHtml(name)}</span>`)
      .join("");
    const processImages = (reconstruction.process_images || [])
      .map((image) => `
        <figure>
          <img src="${image.url}" alt="">
          <figcaption>${escapeHtml(image.step || "")} · ${escapeHtml(image.label || "trace step")}</figcaption>
        </figure>
      `)
      .join("");
    const card = document.createElement("article");
    card.className = "aristotel-card";
    card.innerHTML = `
      <div class="aristotel-card-top">
        <span class="aristotel-index">#${sample.sample_index}</span>
        <strong>${escapeHtml(sample.damage_recipe)}</strong>
        <span class="aristotel-label">${escapeHtml(sample.trust_label)}</span>
        <button class="sample-detail-button" type="button">inspect</button>
      </div>
      <div class="aristotel-visual-block">
        <div class="visual-block-title">
          <strong>Damage pass</strong>
          <span>${formatPercent(sample.changed_pixel_ratio)} changed</span>
        </div>
        <div class="aristotel-image-pair">
          <figure>
            <img src="${sample.original_url}" alt="">
            <figcaption>thresholded source · ${escapeHtml(sample.label)}</figcaption>
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
      </div>
      <div class="reconstruction-preview">
        <div class="reconstruction-title">
          <strong>Defense process</strong>
          <span>${escapeHtml(reconstruction.reconstruction_status || "not run")}</span>
        </div>
        <div class="reconstruction-process-strip">
          ${processImages || '<div class="process-empty">No process images emitted.</div>'}
        </div>
        <div class="reconstruction-metrics">
          <span>candidates ${escapeHtml(reconstruction.candidate_count ?? 0)}</span>
          <span>accepted ${escapeHtml(reconstruction.accepted_count ?? 0)}</span>
          <span>rejected ${escapeHtml(reconstruction.rejected_count ?? 0)}</span>
          <span>${escapeHtml(reconstruction.selected_hypothesis_id || "h0_original")}</span>
          <span>${reconstruction.recognition_bypassed_for_ui ? "RF skipped in UI" : "RF checked"}</span>
        </div>
        <div class="reconstruction-subtitle">Stage plan</div>
        ${renderStageTimeline(reconstruction)}
        <div class="reconstruction-subtitle">Defense tools</div>
        ${renderDefenseToolSummary(reconstruction)}
        <div class="reconstruction-subtitle">Generated hypotheses</div>
        ${renderHypothesisMiniCards(reconstruction)}
        ${renderLineRemovalSequences(reconstruction, { compact: true })}
        <div class="reconstruction-defense-row">
          <strong>routed</strong>
          ${allowedDefenses || "<span>none</span>"}
        </div>
        <div class="reconstruction-defense-row muted">
          <strong>unsupported</strong>
          ${unsupportedDefenses || "<span>none</span>"}
        </div>
        <div class="reconstruction-defense-row muted">
          <strong>signals</strong>
          ${damageReasons || "<span>no topology signal</span>"}
        </div>
        <div class="defense-preview compact">
          <strong>${escapeHtml(sample.defense_preview?.currently_available_tool || "diagnosis")}</strong>
          <p>${escapeHtml(sample.defense_preview?.note || "")}</p>
          <div class="defense-cycle">${cycle}</div>
        </div>
      </div>
      <details class="aristotel-details">
        <summary>metadata</summary>
        <pre>${escapeHtml(JSON.stringify({
          metadata: sample.metadata,
          reconstruction_preview: reconstruction,
        }, null, 2))}</pre>
      </details>
    `;
    card.querySelector(".sample-detail-button").addEventListener("click", () => {
      openSampleDetail(sample);
    });
    card.querySelectorAll(".aristotel-visual-block, .reconstruction-preview").forEach(
      (block) => {
        block.addEventListener("click", () => openSampleDetail(sample));
      },
    );
    elements.aristotelSamples.appendChild(card);
  });
}

function sampleDetailImages(sample) {
  const reconstruction = sample.reconstruction_preview || {};
  const process = (reconstruction.process_images || [])
    .filter((image) => image.step !== "00_damaged");
  return [
    {
      step: "thresholded_source",
      label: `thresholded source · ${sample.label}`,
      url: sample.original_url,
    },
    {
      step: "damaged",
      label: `damaged · ${formatPercent(sample.changed_pixel_ratio)}`,
      url: sample.damaged_url,
    },
    ...process,
  ].filter((image) => image.url);
}

function renderSampleDetailHero(image) {
  const hero = elements.sampleDetailContent.querySelector(".sample-detail-hero");
  const caption = elements.sampleDetailContent.querySelector(".sample-detail-hero-caption");
  if (!hero || !caption || !image) return;
  resetDetailZoom();
  hero.src = image.url;
  caption.textContent = `${image.step || "step"} · ${image.label || "image"}`;
}

function applyDetailTransform() {
  const hero = elements.sampleDetailContent.querySelector(".sample-detail-hero");
  const zoomReset = elements.sampleDetailContent.querySelector(".detail-zoom-reset");
  const heroWrap = elements.sampleDetailContent.querySelector(".sample-detail-hero-wrap");
  if (!hero || !zoomReset || !heroWrap) return;
  hero.style.transform = (
    `translate(${state.detailPanX}px, ${state.detailPanY}px) `
    + `scale(${state.detailZoom})`
  );
  zoomReset.textContent = `${Math.round(state.detailZoom * 100)}%`;
  heroWrap.classList.toggle("zoomed", state.detailZoom > 1);
}

function resetDetailZoom() {
  state.detailZoom = 1;
  state.detailPanX = 0;
  state.detailPanY = 0;
  applyDetailTransform();
}

function setDetailZoom(nextZoom) {
  state.detailZoom = Math.min(8, Math.max(0.5, nextZoom));
  if (state.detailZoom <= 1) {
    state.detailPanX = 0;
    state.detailPanY = 0;
  }
  applyDetailTransform();
}

function wireSampleDetailZoom() {
  const heroWrap = elements.sampleDetailContent.querySelector(".sample-detail-hero-wrap");
  const zoomIn = elements.sampleDetailContent.querySelector(".detail-zoom-in");
  const zoomOut = elements.sampleDetailContent.querySelector(".detail-zoom-out");
  const zoomReset = elements.sampleDetailContent.querySelector(".detail-zoom-reset");
  if (!heroWrap || !zoomIn || !zoomOut || !zoomReset) return;

  zoomIn.addEventListener("click", () => setDetailZoom(state.detailZoom * 1.25));
  zoomOut.addEventListener("click", () => setDetailZoom(state.detailZoom / 1.25));
  zoomReset.addEventListener("click", resetDetailZoom);
  heroWrap.addEventListener("dblclick", resetDetailZoom);
  heroWrap.addEventListener("wheel", (event) => {
    event.preventDefault();
    setDetailZoom(
      state.detailZoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15),
    );
  }, { passive: false });

  heroWrap.addEventListener("pointerdown", (event) => {
    if (state.detailZoom <= 1) return;
    state.detailDragging = true;
    state.detailDragX = event.clientX - state.detailPanX;
    state.detailDragY = event.clientY - state.detailPanY;
    heroWrap.classList.add("dragging");
    heroWrap.setPointerCapture(event.pointerId);
  });

  heroWrap.addEventListener("pointermove", (event) => {
    if (!state.detailDragging) return;
    state.detailPanX = event.clientX - state.detailDragX;
    state.detailPanY = event.clientY - state.detailDragY;
    applyDetailTransform();
  });

  function stopDetailDrag(event) {
    state.detailDragging = false;
    heroWrap.classList.remove("dragging");
    if (event.pointerId !== undefined && heroWrap.hasPointerCapture(event.pointerId)) {
      heroWrap.releasePointerCapture(event.pointerId);
    }
  }

  heroWrap.addEventListener("pointerup", stopDetailDrag);
  heroWrap.addEventListener("pointercancel", stopDetailDrag);
}

function openSampleDetail(sample) {
  const reconstruction = sample.reconstruction_preview || {};
  const resultJsonPath = sampleResultJsonPath(sample);
  const images = sampleDetailImages(sample);
  const firstImage = images[0] || {};
  const operationChips = (sample.operations || [])
    .map((operation) => {
      const name = operation.operation || operation.type || "operation";
      return `<span title="${escapeHtml(JSON.stringify(operation))}">${escapeHtml(name)}</span>`;
    })
    .join("");
  const processThumbs = images
    .map((image, index) => `
      <button
        class="sample-detail-thumb ${index === 0 ? "active" : ""}"
        type="button"
        data-image-index="${index}"
      >
        <img src="${image.url}" alt="">
        <span>${escapeHtml(image.step || "step")}</span>
      </button>
    `)
    .join("");
  const routed = (reconstruction.allowed_defense_types || [])
    .map((name) => `<span>${escapeHtml(name)}</span>`)
    .join("");
  const unsupported = (reconstruction.unsupported_defense_types || [])
    .map((name) => `<span>${escapeHtml(name)}</span>`)
    .join("");

  elements.sampleDetailContent.innerHTML = `
    <div class="sample-detail-layout">
      <section class="sample-detail-gallery">
        <div class="sample-detail-hero-wrap">
          <img class="sample-detail-hero" src="${firstImage.url || ""}" alt="">
          <div class="detail-zoom-controls" aria-label="Detail image zoom controls">
            <button class="detail-zoom-out" type="button" title="Zoom out">−</button>
            <button class="detail-zoom-reset" type="button" title="Reset zoom">100%</button>
            <button class="detail-zoom-in" type="button" title="Zoom in">+</button>
          </div>
          <span class="sample-detail-hero-caption">
            ${escapeHtml(firstImage.step || "image")} · ${escapeHtml(firstImage.label || "")}
          </span>
        </div>
        <div class="sample-detail-thumbs">
          ${processThumbs}
        </div>
      </section>

      <section class="sample-detail-info">
        <p class="eyebrow">ARISTOTEL SAMPLE</p>
        <h2 id="sample-detail-title">${escapeHtml(sample.damage_recipe)}</h2>
        <p class="sample-detail-subtitle">
          ${escapeHtml(sample.label)} · ${escapeHtml(sample.trust_label)}
        </p>
        <div class="sample-detail-score-grid">
          <span><b>${formatPercent(sample.changed_pixel_ratio)}</b><small>changed</small></span>
          <span><b>${sample.changed_pixel_count}</b><small>pixels</small></span>
          <span><b>${Number(sample.severity || 0).toFixed(2)}</b><small>severity</small></span>
          <span><b>${escapeHtml(reconstruction.accepted_count ?? 0)}</b><small>accepted</small></span>
        </div>
        <div class="sample-detail-chip-row">
          ${operationChips || "<span>no operation metadata</span>"}
        </div>
        <div class="sample-detail-defense">
          <strong>routed defenses</strong>
          <div>${routed || "<span>none</span>"}</div>
        </div>
        <div class="sample-detail-defense">
          <strong>stage plan</strong>
          ${renderStageTimeline(reconstruction)}
        </div>
        <div class="sample-detail-defense">
          <strong>tool states</strong>
          ${renderDefenseToolSummary(reconstruction)}
        </div>
        ${renderLineRemovalSequences(reconstruction)}
        <div class="sample-detail-defense muted">
          <strong>unsupported defenses</strong>
          <div>${unsupported || "<span>none</span>"}</div>
        </div>
        ${renderHypothesisWorkbench(reconstruction)}
        <details class="sample-detail-json" open>
          <summary>
            <span>full sample contract</span>
            <span class="sample-json-actions">
              <button
                class="sample-json-filename-copy"
                type="button"
                data-json-filename="${escapeHtml(resultJsonPath)}"
              >copy json_filename</button>
              <button class="sample-json-copy" type="button">copy JSON</button>
            </span>
          </summary>
          <pre data-json-copy-source>${escapeHtml(JSON.stringify(sample, null, 2))}</pre>
        </details>
      </section>
    </div>
  `;

  elements.sampleDetailContent.querySelectorAll(".sample-detail-thumb").forEach((button) => {
    button.addEventListener("click", () => {
      const image = images[Number(button.dataset.imageIndex)];
      elements.sampleDetailContent.querySelectorAll(".sample-detail-thumb").forEach(
        (item) => item.classList.remove("active"),
      );
      button.classList.add("active");
      renderSampleDetailHero(image);
    });
  });

  elements.sampleDetailContent.querySelector(".sample-json-copy")?.addEventListener(
    "click",
    async (event) => {
      event.preventDefault();
      event.stopPropagation();

      const button = event.currentTarget;
      const source = elements.sampleDetailContent.querySelector("[data-json-copy-source]");
      const jsonText = source?.textContent || "";

      if (!jsonText.trim()) {
        showToast("No JSON found to copy.", true);
        return;
      }

      try {
        await copyTextToClipboard(jsonText);
        button.textContent = "copied";
        showToast("Aristotel JSON copied.");
        window.setTimeout(() => {
          button.textContent = "copy JSON";
        }, 1400);
      } catch (error) {
        showToast(`Copy failed: ${error.message}`, true);
      }
    },
  );

  elements.sampleDetailContent.querySelector(".sample-json-filename-copy")?.addEventListener(
    "click",
    async (event) => {
      event.preventDefault();
      event.stopPropagation();

      const button = event.currentTarget;
      const filename = button.dataset.jsonFilename || "";

      if (!filename.trim()) {
        showToast("No JSON filename found to copy.", true);
        return;
      }

      try {
        await copyTextToClipboard(filename);
        button.textContent = "copied";
        showToast("JSON filename copied.");
        window.setTimeout(() => {
          button.textContent = "copy json_filename";
        }, 1400);
      } catch (error) {
        showToast(`Copy failed: ${error.message}`, true);
      }
    },
  );

  resetDetailZoom();
  wireSampleDetailZoom();
  elements.sampleDetail.classList.add("visible");
  elements.sampleDetail.setAttribute("aria-hidden", "false");
}

function closeSampleDetail() {
  resetDetailZoom();
  elements.sampleDetail.classList.remove("visible");
  elements.sampleDetail.setAttribute("aria-hidden", "true");
}

async function loadAristotelPreview() {
  elements.aristotelSummary.textContent = "Asking Aristotel for 10 samples...";
  elements.aristotelSamples.innerHTML = "";
  elements.refreshAristotel.disabled = true;
  try {
    const payload = await requestJson(
      `/api/aristotel-preview?limit=10&refresh=${Date.now()}`,
    );
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

function setScrilogDirty(isDirty) {
  state.scrilogDirty = isDirty;
  if (!elements.scrilogSaveState) return;
  elements.scrilogSaveState.textContent = isDirty ? "UNSAVED CHANGES" : "SAVED";
  elements.scrilogSaveState.classList.toggle("dirty", isDirty);
}

function scrilogDefaultValue(field) {
  return field.type === "bool" ? false : 0;
}

function renderScrilogFields(schema, values = {}) {
  elements.scrilogFields.innerHTML = "";
  let activeGroup = "";
  schema.forEach((field) => {
    if (field.group && field.group !== activeGroup) {
      activeGroup = field.group;
      const heading = document.createElement("div");
      heading.className = "scrilog-field-group";
      heading.textContent = activeGroup;
      elements.scrilogFields.appendChild(heading);
    }
    const wrapper = document.createElement("div");
    wrapper.className = `scrilog-field scrilog-field-${field.type}`;
    const value = values[field.name] ?? scrilogDefaultValue(field);

    if (field.type === "bool") {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `scrilog-toggle ${value ? "active" : ""}`;
      button.dataset.field = field.name;
      button.dataset.value = value ? "true" : "false";
      button.setAttribute("aria-pressed", value ? "true" : "false");
      button.innerHTML = `
        <span>${escapeHtml(field.label)} <em class="importance-${escapeHtml(field.importance)}">${escapeHtml(field.importance)}</em></span>
        <strong>${value ? "ON" : "OFF"}</strong>
      `;
      button.addEventListener("click", () => {
        const nextValue = button.dataset.value !== "true";
        button.dataset.value = nextValue ? "true" : "false";
        button.setAttribute("aria-pressed", nextValue ? "true" : "false");
        button.classList.toggle("active", nextValue);
        button.querySelector("strong").textContent = nextValue ? "ON" : "OFF";
        setScrilogDirty(true);
      });
      wrapper.appendChild(button);
    } else {
      const label = document.createElement("label");
      label.innerHTML = `<span>${escapeHtml(field.label)} <em class="importance-${escapeHtml(field.importance)}">${escapeHtml(field.importance)}</em></span>`;
      const input = document.createElement("input");
      input.type = "number";
      input.dataset.field = field.name;
      input.value = value;
      input.min = field.min ?? 0;
      input.max = field.max ?? 1000000;
      input.step = 1;
      input.placeholder = "0";
      input.addEventListener("input", () => setScrilogDirty(true));
      label.appendChild(input);
      wrapper.appendChild(label);
    }
    elements.scrilogFields.appendChild(wrapper);
  });
}

function collectScrilogValues() {
  const values = {};
  elements.scrilogFields.querySelectorAll("[data-field]").forEach((control) => {
    values[control.dataset.field] = control.classList.contains("scrilog-toggle")
      ? control.dataset.value === "true"
      : Number.parseInt(control.value || "0", 10);
  });
  return values;
}

function renderScrilogWorkspace(payload) {
  state.scrilogWorkspace = payload;
  state.scrilogLoaded = true;
  elements.scrilogSavedCount.textContent = payload.annotation_count || 0;
  elements.scrilogOutputPath.textContent = payload.output_path || "ScriLog output unavailable";

  if (payload.status !== "completed" || !payload.sample) {
    elements.scrilogStatus.textContent = "No Matenadata glyphs found for this selection.";
    elements.scrilogImage.removeAttribute("src");
    elements.scrilogFields.innerHTML = "";
    return;
  }

  if (elements.scrilogClassFilter.options.length <= 1) {
    payload.class_labels.forEach((classLabel) => {
      const option = document.createElement("option");
      option.value = classLabel;
      option.textContent = `Class ${classLabel}`;
      elements.scrilogClassFilter.appendChild(option);
    });
  }
  elements.scrilogClassFilter.value = payload.class_filter || "";
  state.scrilogIndex = payload.index;
  state.scrilogClassFilter = payload.class_filter || "";

  const annotation = payload.sample.annotation || {};
  elements.scrilogIndex.value = payload.index + 1;
  elements.scrilogIndex.max = payload.total;
  elements.scrilogTotal.textContent = `/ ${payload.total.toLocaleString()}`;
  elements.scrilogPrevious.disabled = payload.index <= 0;
  elements.scrilogNext.disabled = payload.index >= payload.total - 1;
  elements.scrilogClassBadge.textContent = `CLASS ${payload.sample.class_label}`;
  elements.scrilogSourceId.textContent = payload.sample.source_id;
  elements.scrilogImageName.textContent = payload.sample.image_name;
  elements.scrilogImage.src = payload.sample.url;
  elements.scrilogNotes.value = annotation.notes || "";
  renderScrilogFields(payload.schema, annotation.expected_signature || {});
  elements.scrilogStatus.textContent = (
    `Glyph ${payload.index + 1} of ${payload.total.toLocaleString()} · `
    + `${payload.annotation_count.toLocaleString()} annotations saved · `
    + `${payload.global_total.toLocaleString()} total source glyphs`
  );
  setScrilogDirty(false);
  elements.scrilogSaveState.textContent = annotation.contract_status === "needs_spatial_review"
    ? "REVIEW SPATIAL FIELDS"
    : annotation.expected_signature ? "SAVED" : "NEW";
  elements.scrilogSaveState.classList.toggle(
    "dirty",
    annotation.contract_status === "needs_spatial_review",
  );
}

async function loadScrilogWorkspace(index = state.scrilogIndex) {
  elements.scrilogStatus.textContent = "Reading the Matenadata sequence...";
  const params = new URLSearchParams({
    index: Math.max(0, index).toString(),
    class: state.scrilogClassFilter,
  });
  try {
    renderScrilogWorkspace(await requestJson(`/api/scrilog-workspace?${params}`));
  } catch (error) {
    elements.scrilogStatus.textContent = error.message;
    showToast(error.message, true);
  }
}

async function saveScrilogAnnotation({ advanceClass = false } = {}) {
  const sample = state.scrilogWorkspace?.sample;
  if (!sample) return;
  try {
    const result = await requestJson("/api/scrilog-annotation", {
      method: "POST",
      body: JSON.stringify({
        source_id: sample.source_id,
        values: collectScrilogValues(),
        notes: elements.scrilogNotes.value,
      }),
    });
    elements.scrilogSavedCount.textContent = result.annotation_count;
    setScrilogDirty(false);
    showToast(`ScriLog saved ${sample.source_id}`);
    if (advanceClass) {
      const classLabels = state.scrilogWorkspace.class_labels || [];
      const currentClass = sample.class_label;
      const currentClassIndex = classLabels.indexOf(currentClass);
      const nextClass = classLabels[currentClassIndex + 1];
      if (nextClass !== undefined) {
        state.scrilogClassFilter = nextClass;
        elements.scrilogClassFilter.value = nextClass;
        await loadScrilogWorkspace(0);
      } else {
        showToast("Last ScriLog class reached.");
      }
    }
  } catch (error) {
    elements.scrilogStatus.textContent = `Save failed: ${error.message}`;
    showToast(error.message, true);
  }
}

async function exportScrilogJson() {
  try {
    const payload = await requestJson("/api/scrilog-export");
    const blob = new Blob(
      [JSON.stringify(payload, null, 2) + "\n"],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "scrilog_annotations.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showToast("ScriLog JSON exported.");
  } catch (error) {
    showToast(error.message, true);
  }
}

function scrististicsValueLabel(value) {
  return Number.isInteger(value) ? value.toString() : value.toFixed(1);
}

function renderScrististicsCurve(points) {
  if (!points.length) {
    return '<div class="scrististics-empty">No numeric observations for this attribute.</div>';
  }
  const width = 760;
  const height = 300;
  const padding = { left: 58, right: 24, top: 24, bottom: 48 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = points.map((point) => Number(point.value));
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const maximumPercent = Math.max(...points.map((point) => point.percent), 1);
  const yMaximum = Math.max(10, Math.ceil(maximumPercent / 10) * 10);
  const xPosition = (value, index) => {
    if (maxValue === minValue) return padding.left + plotWidth / 2;
    return padding.left + ((value - minValue) / (maxValue - minValue)) * plotWidth;
  };
  const yPosition = (percent) => (
    padding.top + plotHeight - (percent / yMaximum) * plotHeight
  );
  const coordinates = points.map((point, index) => ({
    ...point,
    x: xPosition(Number(point.value), index),
    y: yPosition(point.percent),
  }));
  const linePath = coordinates.map((point, index) => (
    `${index ? "L" : "M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`
  )).join(" ");
  const baseY = padding.top + plotHeight;
  const areaPath = (
    `M${coordinates[0].x.toFixed(1)},${baseY.toFixed(1)} `
    + `${linePath} L${coordinates.at(-1).x.toFixed(1)},${baseY.toFixed(1)} Z`
  );
  const grid = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const y = padding.top + plotHeight - ratio * plotHeight;
    const label = Math.round(ratio * yMaximum);
    return `
      <line class="scrististics-grid" x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}"></line>
      <text class="scrististics-y-label" x="${padding.left - 10}" y="${y + 3}" text-anchor="end">${label}%</text>
    `;
  }).join("");
  const stems = coordinates.map((point) => `
    <line class="scrististics-stem" x1="${point.x}" y1="${baseY}" x2="${point.x}" y2="${point.y}"></line>
  `).join("");
  const dots = coordinates.map((point) => `
    <g class="scrististics-point">
      <circle cx="${point.x}" cy="${point.y}" r="5"></circle>
      <title>value ${scrististicsValueLabel(Number(point.value))}: ${point.percent}% (${point.count} samples)</title>
    </g>
  `).join("");
  const labels = coordinates.map((point, index) => {
    if (coordinates.length > 16 && index % Math.ceil(coordinates.length / 12) !== 0) return "";
    return `<text class="scrististics-x-label" x="${point.x}" y="${height - 18}" text-anchor="middle">${scrististicsValueLabel(Number(point.value))}</text>`;
  }).join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Empirical topology distribution">
      <defs>
        <linearGradient id="scrististics-area-gradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#47b58b" stop-opacity="0.42"></stop>
          <stop offset="100%" stop-color="#47b58b" stop-opacity="0.02"></stop>
        </linearGradient>
      </defs>
      ${grid}
      <line class="scrististics-axis" x1="${padding.left}" y1="${baseY}" x2="${width - padding.right}" y2="${baseY}"></line>
      ${stems}
      <path class="scrististics-area" d="${areaPath}"></path>
      <path class="scrististics-curve" d="${linePath}"></path>
      ${dots}
      ${labels}
    </svg>
  `;
}

function renderScrististicsGeometry(summary) {
  if (!summary || summary.mean === null || summary.mean === undefined) {
    return '<div class="scrististics-empty">No sizing observations for this attribute.</div>';
  }
  const min = Number(summary.min);
  const max = Number(summary.max);
  const markers = [
    ["p10", "P10", summary.p10],
    ["median", "Median", summary.median],
    ["mean", "Mean", summary.mean],
    ["p90", "P90", summary.p90],
  ].filter(([, , value]) => value !== null && value !== undefined && !Number.isNaN(Number(value)));
  const position = (value) => {
    if (max === min) return 50;
    return Math.max(0, Math.min(100, ((Number(value) - min) / (max - min)) * 100));
  };
  const markerMarkup = markers.map(([key, label, value]) => `
    <span class="scrististics-geometry-marker ${key}" style="left:${position(value).toFixed(2)}%">
      <i></i>
      <strong>${escapeHtml(label)}</strong>
      <small>${scrististicsValueLabel(Number(value))}</small>
    </span>
  `).join("");
  return `
    <div class="scrististics-geometry-card">
      <div class="scrististics-geometry-scale">
        <span>${scrististicsValueLabel(min)}</span>
        <span>${scrististicsValueLabel(max)}</span>
      </div>
      <div class="scrististics-geometry-track">
        <div class="scrististics-geometry-core" style="
          left:${position(summary.p10).toFixed(2)}%;
          width:${Math.max(1, position(summary.p90) - position(summary.p10)).toFixed(2)}%;
        "></div>
        ${markerMarkup}
      </div>
      <p>
        Geometry profile is measured from the binary ink bounding box. Use this
        for crop plausibility, not as a hard letter identity rule.
      </p>
    </div>
  `;
}

function renderScrististicsDistribution(payload) {
  state.scrististicsLoaded = true;
  state.scrististicsClass = payload.selected_class.class_id;
  state.scrististicsFeature = payload.selected_feature.name;

  const previousClass = elements.scrististicsClass.value;
  const previousFeature = elements.scrististicsFeature.value;
  elements.scrististicsClass.innerHTML = payload.classes.map((row) => (
    `<option value="${escapeHtml(row.class_id)}">${escapeHtml(row.class_id)} · ${escapeHtml(row.label)}</option>`
  )).join("");
  elements.scrististicsFeature.innerHTML = payload.features.map((row) => (
    `<option value="${escapeHtml(row.name)}">${escapeHtml(row.label)}</option>`
  )).join("");
  elements.scrististicsClass.value = payload.selected_class.class_id || previousClass;
  elements.scrististicsFeature.value = payload.selected_feature.name || previousFeature;

  elements.scrististicsProfilePath.textContent = payload.profile_path;
  elements.scrististicsDatasetSummary.textContent = (
    `${payload.dataset_sample_count.toLocaleString()} traced glyphs · `
    + `${payload.dataset_class_count} Armenian classes · `
    + `${Math.round(payload.elapsed_seconds / 60)} min mining run`
  );
  elements.scrististicsClassBadge.textContent = (
    `CLASS ${payload.selected_class.class_id} · ${payload.selected_class.label}`
  );
  elements.scrististicsChartTitle.textContent = payload.selected_feature.label;
  elements.scrististicsMode.textContent = (
    payload.selected_feature.kind === "geometry"
      ? `MEAN ${scrististicsValueLabel(Number(payload.selected_feature.summary?.mean ?? 0))}`
      : `MODE ${payload.selected_feature.most_common_value ?? "—"}`
  );
  elements.scrististicsChart.innerHTML = payload.selected_feature.kind === "geometry"
    ? renderScrististicsGeometry(payload.selected_feature.summary)
    : renderScrististicsCurve(payload.selected_feature.points || []);

  const geometrySummary = payload.selected_feature.summary || {};
  elements.scrististicsKeyMetrics.innerHTML = payload.selected_feature.kind === "geometry"
    ? `
      <div><strong>${payload.selected_class.sample_count.toLocaleString()}</strong><span>class samples</span></div>
      <div><strong>${scrististicsValueLabel(Number(geometrySummary.mean ?? 0))}</strong><span>mean</span></div>
      <div><strong>${scrististicsValueLabel(Number(geometrySummary.median ?? 0))}</strong><span>median</span></div>
      <div><strong>${scrististicsValueLabel(Number(geometrySummary.stdev ?? 0))}</strong><span>stdev</span></div>
    `
    : `
      <div><strong>${payload.selected_class.sample_count.toLocaleString()}</strong><span>class samples</span></div>
      <div><strong>${payload.representative.support_percent}%</strong><span>joint-mode support</span></div>
      <div><strong>${payload.selected_feature.points.length}</strong><span>observed values</span></div>
      <div><strong>${escapeHtml(payload.selected_feature.importance)}</strong><span>feature tier</span></div>
    `;
  elements.scrististicsVariants.innerHTML = payload.variants.length
    ? payload.variants.map((variant) => `
        <article>
          <span>V${variant.rank}</span>
          <div>
            <strong>${escapeHtml(variant.source_id || "unknown source")}</strong>
            <small>${variant.support_percent}% support · ${variant.count.toLocaleString()} glyphs</small>
          </div>
        </article>
      `).join("")
    : '<div class="scrististics-empty">No joint variants available.</div>';
}

async function loadScrististicsDistribution() {
  elements.scrististicsChart.innerHTML = (
    '<div class="scrististics-empty">Reading empirical topology profiles...</div>'
  );
  const params = new URLSearchParams({
    class: state.scrististicsClass,
    feature: state.scrististicsFeature,
  });
  try {
    renderScrististicsDistribution(
      await requestJson(`/api/scrististics-distribution?${params}`),
    );
  } catch (error) {
    elements.scrististicsChart.innerHTML = (
      `<div class="scrististics-empty error">${escapeHtml(error.message)}</div>`
    );
    showToast(error.message, true);
  }
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
  document.querySelector(".console-panel")?.scrollIntoView({
    behavior: "smooth",
    block: "end",
  });
  try {
    const payload = await requestJson("/api/run", {
      method: "POST",
      body: JSON.stringify({ command }),
    });
    renderProcess(payload.process);
    showToast(`${command} launched`);
    window.setTimeout(() => {
      elements.consoleOutput.scrollTop = elements.consoleOutput.scrollHeight;
    }, 80);
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
    if (button.dataset.decisionCanvas === "true") {
      document.querySelector(".decision-canvas-panel")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
    if (state.selectedStage === "n04") {
      document.querySelector(".printed-context-panel")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  });
});

elements.stopButton.addEventListener("click", stopCommand);
elements.hardRefreshUi.addEventListener("click", hardRefreshUi);
elements.refreshArtifacts.addEventListener("click", () => {
  loadOverview({ refreshArtifacts: true });
});
elements.refreshPrintedContext.addEventListener("click", loadPrintedContextCanvas);
elements.printedContextImage.addEventListener("load", renderPrintedContextBoxes);
elements.refreshDecisionCanvas.addEventListener("click", loadDecisionCanvas);
elements.decisionCanvasImage.addEventListener("load", renderDecisionCanvasBoxes);
window.addEventListener("resize", () => {
  renderPrintedContextBoxes();
  renderDecisionCanvasBoxes();
});
elements.refreshAristotel.addEventListener("click", loadAristotelPreview);
elements.refreshTraining.addEventListener("click", loadTrainingOverview);
elements.runMinosTraining.addEventListener("click", () => runCommand("train"));
elements.scrilogPrevious.addEventListener("click", () => {
  loadScrilogWorkspace(state.scrilogIndex - 1);
});
elements.scrilogNext.addEventListener("click", () => {
  loadScrilogWorkspace(state.scrilogIndex + 1);
});
elements.scrilogIndex.addEventListener("change", () => {
  loadScrilogWorkspace(Math.max(0, Number.parseInt(elements.scrilogIndex.value || "1", 10) - 1));
});
elements.scrilogClassFilter.addEventListener("change", () => {
  state.scrilogClassFilter = elements.scrilogClassFilter.value;
  loadScrilogWorkspace(0);
});
elements.scrilogSave.addEventListener("click", () => saveScrilogAnnotation());
elements.scrilogSaveNext.addEventListener("click", () => saveScrilogAnnotation({ advanceClass: true }));
elements.scrilogSidebarSave.addEventListener("click", () => saveScrilogAnnotation());
elements.scrilogExport.addEventListener("click", exportScrilogJson);
elements.scrilogSidebarExport.addEventListener("click", exportScrilogJson);
elements.scrilogReset.addEventListener("click", () => {
  renderScrilogFields(state.scrilogWorkspace?.schema || [], {});
  elements.scrilogNotes.value = "";
  setScrilogDirty(true);
});
elements.scrilogNotes.addEventListener("input", () => setScrilogDirty(true));
elements.scrististicsClass.addEventListener("change", () => {
  state.scrististicsClass = elements.scrististicsClass.value;
  loadScrististicsDistribution();
});
elements.scrististicsFeature.addEventListener("change", () => {
  state.scrististicsFeature = elements.scrististicsFeature.value;
  loadScrististicsDistribution();
});
elements.scrististicsRefresh.addEventListener("click", () => {
  state.scrististicsLoaded = false;
  loadScrististicsDistribution();
});
elements.sampleDetailBackdrop.addEventListener("click", closeSampleDetail);
elements.sampleDetailClose.addEventListener("click", closeSampleDetail);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && elements.sampleDetail.classList.contains("visible")) {
    closeSampleDetail();
  }
});
document.querySelectorAll(".sidebar-refresh-aristotel").forEach((button) => {
  button.addEventListener("click", loadAristotelPreview);
});
document.querySelectorAll(".sidebar-refresh-training").forEach((button) => {
  button.addEventListener("click", loadTrainingOverview);
});
elements.clearConsole.addEventListener("click", () => {
  elements.consoleOutput.textContent = "";
});
elements.elevatorUp?.addEventListener("click", () => elevatorScroll("up"));
elements.elevatorDown?.addEventListener("click", () => elevatorScroll("down"));
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

document.addEventListener("wheel", (event) => {
  const targetElement = (
    event.target instanceof Element ? event.target : event.target.parentElement
  );
  const horizontalStrip = targetElement?.closest(
    ".reconstruction-process-strip, .stage-timeline, .hypothesis-strip, .hypothesis-detail-images, .line-removal-images, .aristotel-image-pair, .sample-detail-thumbs",
  );
  if (!horizontalStrip) return;

  const canScrollHorizontally = (
    horizontalStrip.scrollWidth > horizontalStrip.clientWidth
  );
  if (!canScrollHorizontally) return;

  event.preventDefault();
  horizontalStrip.scrollLeft += event.deltaY || event.deltaX;
}, { passive: false });

elements.previewViewport.addEventListener("dblclick", resetPreviewZoom);
elements.cropContextImage.addEventListener("load", renderCropContextBox);
window.addEventListener("resize", renderCropContextBox);

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

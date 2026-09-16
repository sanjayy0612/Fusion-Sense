const state = { data: null, index: 0, timer: null, poller: null, followLatest: true };
const $ = (id) => document.getElementById(id);

function percent(value, digits = 1) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function friendly(label) {
  return label ? label.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()) : "Unavailable";
}

function metric(value) {
  return value === undefined || value === null ? "N/A" : percent(value);
}

function timeLabel(value, includeDate = true) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "—";
  return includeDate ? parsed.toLocaleString() : parsed.toLocaleTimeString();
}

function renderStatic(data) {
  const live = data.mode === "live_esp32";
  $("mode-badge").textContent = live ? "LIVE ESP32" : data.mode === "mentor_demo" ? "LABELED REPLAY" : "RECORDED ESP32";
  $("mode-badge").classList.toggle("live", live);
  $("generated-at").textContent = `Generated ${new Date(data.generated_at_utc).toLocaleString()}`;
  $("disclosure").textContent = data.disclosure;
  $("scope-note").textContent = data.model?.scope_note || "Research prototype.";
  $("footer-source").textContent = data.model?.name || "Local checkpoint output";

  const evaluation = data.evaluation || {};
  $("metric-accuracy").textContent = metric(evaluation.accuracy);
  $("metric-f1").textContent = metric(evaluation.macro_f1);
  $("metric-fall-recall").textContent = metric(evaluation.fall_recall);

  $("control-strip").classList.toggle("live-mode", live);
  $("play").hidden = live;
  $("previous").textContent = live ? "Older" : "Previous";
  $("next").textContent = live ? "Latest" : "Next";

  const contract = data.input_contract || {};
  $("contract-window").textContent = `${contract.window_seconds || 2} seconds`;
  $("contract-imu").textContent = `${(contract.imu?.shape || []).join(" × ")} · m/s² + deg/s`;
  $("contract-camera").textContent = `${(contract.camera_pose?.shape || []).join(" × ")} · pose landmarks`;
  $("contract-conversion").textContent = contract.live_esp32_conversion?.acceleration_formula || "m/s² canonical";
  const unitCheck = contract.dataset_unit_validation;
  $("contract-validation").textContent = unitCheck
    ? `${unitCheck.valid ? "PASS" : "FAIL"} · median |a| ${unitCheck.acceleration_norm_median_mps2} m/s²`
    : "Reported per recorded session";
}

function renderProbabilities(event) {
  const container = $("probabilities");
  container.replaceChildren();
  const probabilities = Object.entries(event.class_probabilities || {});
  if (!probabilities.length) {
    const note = document.createElement("p");
    note.className = "muted empty-state";
    note.textContent = "Prediction withheld until both IMU and full-body camera pose are valid.";
    container.appendChild(note);
    return;
  }
  probabilities
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = `probability-row${label === "stand_to_fall" ? " fall" : ""}`;
      row.innerHTML = `
        <span>${friendly(label)}</span>
        <div class="probability-track"><div class="probability-fill" style="width:${Math.max(0, Math.min(100, value * 100))}%"></div></div>
        <span class="probability-value">${percent(value, 0)}</span>`;
      container.appendChild(row);
    });
}

function renderSensors(event) {
  const names = [
    ["imu", "IMU", event.input?.valid_modalities?.imu, event.sensor_trust?.imu, event.input?.health?.imu],
    ["camera", "ESP32-CAM pose", event.input?.valid_modalities?.camera_pose, event.sensor_trust?.camera, event.input?.health?.camera],
  ];
  const container = $("sensor-list");
  container.replaceChildren();
  names.forEach(([, label, valid, trust = 0, health = 0]) => {
    const row = document.createElement("div");
    row.className = "sensor-row";
    row.innerHTML = `
      <div class="sensor-top">
        <strong>${label}</strong>
        <span class="sensor-state ${valid ? "valid" : "invalid"}">${valid ? `VALID · ${percent(health, 0)} health · ${percent(trust, 0)} trust` : `DEGRADED · ${percent(health, 0)} health`}</span>
      </div>
      <div class="sensor-meter"><span style="width:${valid ? Math.max(2, trust * 100) : 0}%"></span></div>`;
    container.appendChild(row);
  });
}

function renderTable(events, activeIndex) {
  const body = $("events-table");
  body.replaceChildren();
  events.forEach((event, index) => {
    const row = document.createElement("tr");
    if (index === activeIndex) row.className = "active";
    const degraded = event.status === "degraded" || event.status === "sensor_unavailable";
    row.innerHTML = `
      <td>${event.sequence}</td>
      <td>${timeLabel(event.event_timestamp_utc, false)}</td>
      <td>${friendly(event.ground_truth)}</td>
      <td>${friendly(event.predicted_label)}</td>
      <td>${percent(event.fall_probability)}</td>
      <td class="${event.alert ? "status-alert" : degraded ? "status-degraded" : "status-ok"}">${event.alert ? "FALL ALERT" : degraded ? "Degraded" : "Monitoring"}</td>`;
    row.addEventListener("click", () => { state.index = index; pause(); render(); });
    body.appendChild(row);
  });
}

function render() {
  const events = state.data?.events || [];
  if (!events.length) {
    $("alert-title").textContent = state.data?.live?.state === "error" ? "Live input unavailable" : "Waiting for synchronized input";
    $("alert-copy").textContent = state.data?.live?.errors?.join("; ") || "Collecting the first valid two-second IMU and camera window.";
    $("event-position").textContent = "No inference windows yet";
    $("fall-score").textContent = "—";
    return;
  }
  const event = events[state.index];
  const alertPanel = $("alert-panel");
  const degraded = event.status === "degraded" || event.status === "sensor_unavailable";
  alertPanel.classList.toggle("alerting", Boolean(event.alert));
  alertPanel.classList.toggle("degraded", degraded);
  $("alert-kicker").textContent = event.alert ? "ACTION REQUIRED" : degraded ? "INPUT ATTENTION" : "SYSTEM STATUS";
  $("alert-title").textContent = event.alert ? "Fall detected" : degraded ? "Sensor degraded" : "Monitoring";
  $("alert-copy").textContent = event.alert
    ? `Stand-to-fall exceeded the ${(event.fall_threshold * 100).toFixed(0)}% alert threshold. Check the monitored person.`
    : degraded
      ? event.reason || "Prediction withheld because both required inputs are not valid."
      : `No fall alert in this two-second window. Predicted ${friendly(event.predicted_label)}.`;
  $("fall-score").textContent = degraded ? "—" : percent(event.fall_probability);
  $("event-position").textContent = `Event ${state.index + 1} of ${events.length}`;
  $("event-id").textContent = event.event_id;
  $("predicted-label").textContent = friendly(event.predicted_label);
  $("truth-label").textContent = event.ground_truth ? friendly(event.ground_truth) : "Not labeled";
  $("event-time").textContent = timeLabel(event.event_timestamp_utc);
  $("prediction-confidence").textContent = degraded ? "Prediction withheld" : `${percent(event.confidence)} confidence`;
  $("metric-inference").textContent = event.inference_ms === undefined ? "—" : `${Number(event.inference_ms).toFixed(1)} ms`;
  renderProbabilities(event);
  renderSensors(event);
  renderTable(events, state.index);
}

function pause() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $("play").textContent = "Play replay";
}

function play() {
  if (state.timer) { pause(); return; }
  if (state.index >= state.data.events.length - 1) state.index = 0;
  render();
  $("play").textContent = "Pause";
  state.timer = setInterval(() => {
    if (state.index >= state.data.events.length - 1) { pause(); return; }
    state.index += 1;
    render();
  }, 1800);
}

async function load({ preserveIndex = false } = {}) {
  const params = new URLSearchParams(window.location.search);
  const dataFile = params.get("data") || "mentor_demo.json";
  try {
    const response = await fetch(dataFile, { cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    state.data = await response.json();
    const events = state.data.events || [];
    state.index = preserveIndex ? Math.min(state.index, Math.max(0, events.length - 1)) : Math.max(0, events.length - 1);
    renderStatic(state.data);
    render();
    if (state.data.mode === "live_esp32" && !state.poller) {
      state.poller = setInterval(() => load({ preserveIndex: !state.followLatest }), 1000);
    }
  } catch (error) {
    $("alert-title").textContent = "Dashboard data unavailable";
    $("alert-copy").textContent = `${error}. Run scripts/run_fall_pipeline.py, then serve this directory over HTTP.`;
  }
}

$("previous").addEventListener("click", () => {
  pause(); state.followLatest = false; state.index = Math.max(0, state.index - 1); render();
});
$("next").addEventListener("click", () => {
  pause();
  const events = state.data?.events || [];
  if (!events.length) return;
  if (state.data?.mode === "live_esp32") {
    state.followLatest = true;
    state.index = events.length - 1;
  } else {
    state.index = Math.min(events.length - 1, state.index + 1);
  }
  render();
});
$("play").addEventListener("click", play);
load();

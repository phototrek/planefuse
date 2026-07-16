const STORAGE_KEY = "planefuse-guided-setup-v1";
const PLATFORMS = ["macos", "windows"];

const tabs = [...document.querySelectorAll("[data-platform-tab]")];
const panels = [...document.querySelectorAll("[data-platform-panel]")];
const progress = document.querySelector("[data-progress]");
const progressLabel = document.querySelector("[data-progress-label]");
const detectedNote = document.querySelector("[data-detected-note]");
const finish = document.querySelector("[data-finish]");

function readState() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}");
    return {
      platform: PLATFORMS.includes(saved.platform) ? saved.platform : null,
      complete: {
        macos: Array.isArray(saved.complete?.macos) ? saved.complete.macos : [],
        windows: Array.isArray(saved.complete?.windows) ? saved.complete.windows : []
      }
    };
  } catch {
    return { platform: null, complete: { macos: [], windows: [] } };
  }
}

const state = readState();

function detectPlatform() {
  const label = `${navigator.userAgentData?.platform ?? ""} ${navigator.platform ?? ""} ${navigator.userAgent}`;
  return /Win/i.test(label) ? "windows" : "macos";
}

function saveState() {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function currentPanel() {
  return document.querySelector(`[data-platform-panel="${state.platform}"]`);
}

function renderProgress() {
  const panel = currentPanel();
  const steps = [...panel.querySelectorAll("[data-step]")];
  const complete = new Set(state.complete[state.platform]);

  for (const step of steps) {
    const done = complete.has(step.dataset.step);
    step.classList.toggle("is-complete", done);
    const toggle = step.querySelector("[data-toggle-step]");
    if (toggle) {
      toggle.setAttribute("aria-pressed", String(done));
    }
  }

  const count = steps.filter((step) => complete.has(step.dataset.step)).length;
  const percentage = `${(count / steps.length) * 100}%`;
  progress.style.setProperty("--progress", percentage);
  progress.setAttribute("aria-valuenow", String(count));
  progressLabel.textContent = `${count} of ${steps.length}`;
  finish.hidden = count !== steps.length;
}

function selectPlatform(platform, announce = true) {
  state.platform = PLATFORMS.includes(platform) ? platform : "macos";

  for (const tab of tabs) {
    const active = tab.dataset.platformTab === state.platform;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }

  for (const panel of panels) {
    const active = panel.dataset.platformPanel === state.platform;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  }

  if (announce) {
    detectedNote.textContent = state.platform === "windows"
      ? "Showing the Windows instructions."
      : "Showing the Apple-silicon Mac instructions.";
  }

  saveState();
  renderProgress();
}

function setStep(step, complete) {
  const steps = new Set(state.complete[state.platform]);
  if (complete) {
    steps.add(step);
  } else {
    steps.delete(step);
  }
  state.complete[state.platform] = [...steps];
  saveState();
  renderProgress();
}

for (const tab of tabs) {
  tab.addEventListener("click", () => selectPlatform(tab.dataset.platformTab));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) {
      return;
    }
    event.preventDefault();
    const index = tabs.indexOf(tab);
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const next = tabs[(index + direction + tabs.length) % tabs.length];
    selectPlatform(next.dataset.platformTab);
    next.focus();
  });
}

for (const link of document.querySelectorAll("[data-complete-on-click]")) {
  link.addEventListener("click", () => setStep(link.dataset.completeOnClick, true));
}

for (const button of document.querySelectorAll("[data-toggle-step]")) {
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", () => {
    const step = button.dataset.toggleStep;
    const complete = !state.complete[state.platform].includes(step);
    setStep(step, complete);
  });
}

document.querySelector("[data-reset]").addEventListener("click", () => {
  state.complete[state.platform] = [];
  saveState();
  renderProgress();
  document.querySelector("#setup").scrollIntoView({ behavior: "smooth", block: "start" });
});

const detected = detectPlatform();
selectPlatform(state.platform ?? detected, false);
detectedNote.textContent = detected === "windows"
  ? "Windows detected — you can switch if needed."
  : "Mac detected — you can switch if needed.";

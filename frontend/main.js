// --- Main Application Logic ---

const statusDiv = document.getElementById("status");
const authSection = document.getElementById("auth-section");
const appSection = document.getElementById("app-section");
const sessionEndSection = document.getElementById("session-end-section");
const restartBtn = document.getElementById("restartBtn");
const micBtn = document.getElementById("micBtn");
const cameraBtn = document.getElementById("cameraBtn");
const screenBtn = document.getElementById("screenBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const videoPreview = document.getElementById("video-preview");
const videoPlaceholder = document.getElementById("video-placeholder");
const connectBtn = document.getElementById("connectBtn");
const chatLog = document.getElementById("chat-log");
const realtimeOptions = document.getElementById("realtime-options");
const openMultimodalBtn = document.getElementById("openMultimodalBtn");
const multimodalSection = document.getElementById("multimodal-section");
const closeMultimodalBtn = document.getElementById("closeMultimodalBtn");
const recordBtn = document.getElementById("recordBtn");
const multimodalSendBtn = document.getElementById("multimodalSendBtn");
const multimodalTextInput = document.getElementById("multimodalTextInput");
const multimodalChatLog = document.getElementById("multimodal-chat-log");
const multimodalFile = document.getElementById("multimodalFile");
const multimodalPlaceholder = document.getElementById("multimodal-placeholder");
const multimodalImagePreview = document.getElementById("multimodal-image-preview");
const multimodalVideoPreview = document.getElementById("multimodal-video-preview");
const multimodalMediaNote = document.getElementById("multimodal-media-note");
const multimodalModelSelect = document.getElementById("multimodalModelSelect");
const multimodalCameraBtn = document.getElementById("multimodalCameraBtn");
const multimodalMicBtn = document.getElementById("multimodalMicBtn");

// ── Structured Output (itinerary consistency tester) elements ──────────────
const openStructuredBtn = document.getElementById("openStructuredBtn");
const structuredSection = document.getElementById("structured-section");
const closeStructuredBtn = document.getElementById("closeStructuredBtn");
const structuredSendBtn = document.getElementById("structuredSendBtn");
const structuredClearBtn = document.getElementById("structuredClearBtn");
const structuredGeminiSelect = document.getElementById("structuredGeminiSelect");
const structuredQwenSelect = document.getElementById("structuredQwenSelect");
const destinationInput = document.getElementById("destinationInput");
const daysInput = document.getElementById("daysInput");
const preferenceInput = document.getElementById("preferenceInput");
const runsInput = document.getElementById("runsInput");

let multimodalCameraStream = null;
let multimodalMicStream = null;

let currentGeminiMessageDiv = null;
let currentUserMessageDiv = null;
let latestInputTokens = 0;
let latestOutputTokens = 0;
let latestMultimodalInputTokens = 0;
let latestMultimodalOutputTokens = 0;
let activeProvider = "gemini";
let activeFlow = "realtime";
let multimodalRecorder = null;
let multimodalRecordStream = null;
let multimodalVideoChunks = [];
let multimodalVideoBlob = null;
let multimodalUploadedFile = null;
let multimodalObjectUrl = null;
let activeMultimodalModel = "gemini-3.1-flash-lite";
let activeMultimodalMode = "function_call";
let discardMultimodalRecording = false;

let multimodalHistory = [];

// ── Timestamp helpers ────────────────────────────────────────────────────────

function fmtTime(date) {
  const h = String(date.getHours()).padStart(2, "0");
  const m = String(date.getMinutes()).padStart(2, "0");
  const s = String(date.getSeconds()).padStart(2, "0");
  const ms = String(date.getMilliseconds()).padStart(3, "0");
  return `${h}:${m}:${s}.${ms}`;
}

function fmtDelta(ms) {
  return `+${(ms / 1000).toFixed(2)}s`;
}

// ── Inject timestamp CSS once ────────────────────────────────────────────────

(function injectTimestampStyles() {
  const style = document.createElement("style");
  style.textContent = `
    .message {
      position: relative;
      padding-bottom: 18px; /* room for timestamp */
    }
    .msg-ts {
      position: absolute;
      bottom: 3px;
      font-size: 10px;
      line-height: 1;
      opacity: 0.55;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
      letter-spacing: 0.01em;
    }
    .message.user  .msg-ts { right: 6px; }
    .message.gemini .msg-ts { left: 6px; }
  `;
  document.head.appendChild(style);
})();

// ────────────────────────────────────────────────────────────────────────────

function parseModelSelect(value) {
  const [model, mode] = value.split("|");
  return { model, mode: mode || "function_call" };
}

const PROVIDER_LABELS = {
  gemini: "Gemini Live",
  qwen: "Qwen-Omni",
};

const PROVIDER_INTROS = {
  gemini: `System: Introduce yourself as a demo of the Gemini Live API.
       Suggest playing with features like the native audio for accents and multilingual support.
       Keep the intro concise and friendly.`,
  qwen: `System: Introduce yourself as a demo of the Qwen-Omni Realtime API.
       Mention you support natural voice conversation, vision (camera/screen sharing), and multilingual interaction.
       Keep the intro concise and friendly.`,
};

const mediaHandler = new MediaHandler();
const geminiClient = new GeminiClient({
  onOpen: () => {
    const label = PROVIDER_LABELS[activeProvider] || activeProvider;
    statusDiv.textContent = `Connected (${label})`;
    statusDiv.className = "status connected";
    authSection.classList.add("hidden");
    appSection.classList.remove("hidden");
  },
  onMessage: (event) => {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        handleJsonMessage(msg);
      } catch (e) {
        console.error("Parse error:", e);
      }
    } else {
      mediaHandler.playAudio(event.data);
    }
  },
  onClose: (e) => {
    console.log("WS Closed:", e);
    statusDiv.textContent = "Disconnected";
    statusDiv.className = "status disconnected";
    showSessionEnd();
  },
  onError: (e) => {
    console.error("WS Error:", e);
    statusDiv.textContent = "Connection Error";
    statusDiv.className = "status error";
  },
});

function updateTokenDisplay() {
  document.getElementById("inputTokens").textContent = latestInputTokens.toLocaleString();
  document.getElementById("outputTokens").textContent = latestOutputTokens.toLocaleString();
}

function updateMultimodalTokenDisplay() {
  document.getElementById("multimodalInputTokens").textContent =
    latestMultimodalInputTokens.toLocaleString();
  document.getElementById("multimodalOutputTokens").textContent =
    latestMultimodalOutputTokens.toLocaleString();
}

function handleJsonMessage(msg) {
  if (msg.type === "usage_metadata") {
    latestInputTokens = msg.input_tokens || 0;
    latestOutputTokens = msg.output_tokens || 0;
    updateTokenDisplay();
  } else if (msg.type === "interrupted") {
    mediaHandler.stopAudioPlayback();
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "turn_complete") {
    currentGeminiMessageDiv = null;
    currentUserMessageDiv = null;
  } else if (msg.type === "user") {
    if (currentUserMessageDiv) {
      currentUserMessageDiv.querySelector(".msg-body").textContent += msg.text;
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentUserMessageDiv = appendMessage("user", msg.text);
    }
  } else if (msg.type === "gemini") {
    if (currentGeminiMessageDiv) {
      currentGeminiMessageDiv.querySelector(".msg-body").textContent += msg.text;
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentGeminiMessageDiv = appendMessage("gemini", msg.text);
    }
  }
}

function appendMessage(type, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;

  const body = document.createElement("span");
  body.className = "msg-body";
  body.textContent = text;
  msgDiv.appendChild(body);

  const ts = document.createElement("span");
  ts.className = "msg-ts";
  ts.textContent = fmtTime(new Date());
  msgDiv.appendChild(ts);

  chatLog.appendChild(msgDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msgDiv;
}

/**
 * Append a message bubble to the multimodal chat log.
 * @param {string} type  "user" | "gemini"
 * @param {string} text
 * @param {object} [opts]
 * @param {Date}   [opts.sentAt]      — timestamp of the user send (to compute delta on model bubble)
 * @param {boolean}[opts.pending]     — if true, timestamp shows "…" until finaliseMultimodalBubble() is called
 * @returns {{ div: HTMLElement, finalize: (text:string, receivedAt:Date) => void }}
 */
function appendMultimodalMessage(type, text, opts = {}) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;

  const body = document.createElement("span");
  body.className = "msg-body";
  body.textContent = text;
  msgDiv.appendChild(body);

  const ts = document.createElement("span");
  ts.className = "msg-ts";

  const now = new Date();

  if (opts.pending) {
    ts.textContent = `${fmtTime(now)} · ⏳`;
  } else {
    ts.textContent = fmtTime(now);
  }

  msgDiv.appendChild(ts);
  multimodalChatLog.appendChild(msgDiv);
  multimodalChatLog.scrollTop = multimodalChatLog.scrollHeight;

  // Finalize: call this when the response arrives to fill in real text + latency
  function finalize(finalText, receivedAt, sentAt) {
    body.textContent = finalText;
    const delta = sentAt ? fmtDelta(receivedAt - sentAt) : "";
    ts.textContent = `${fmtTime(receivedAt)}${delta ? " · " + delta : ""}`;
    multimodalChatLog.scrollTop = multimodalChatLog.scrollHeight;
  }

  return { div: msgDiv, finalize };
}

function updateFlowUI() {
  const selected = document.querySelector('input[name="flow"]:checked');
  activeFlow = selected ? selected.value : "realtime";

  const isRealtime = activeFlow === "realtime";
  const isMultimodal = activeFlow === "multimodal";
  const isStructured = activeFlow === "structured";

  realtimeOptions.classList.toggle("hidden", !isRealtime);
  connectBtn.classList.toggle("hidden", !isRealtime);
  openMultimodalBtn.classList.toggle("hidden", !isMultimodal);
  openStructuredBtn.classList.toggle("hidden", !isStructured);

  if (isRealtime) {
    statusDiv.textContent = "Disconnected";
    statusDiv.className = "status disconnected";
  } else {
    statusDiv.textContent = isStructured ? "Structured Output" : "Multimodal Preview";
    statusDiv.className = "status connected";
  }
}

document.querySelectorAll('input[name="flow"]').forEach((input) => {
  input.addEventListener("change", updateFlowUI);
});

function openMultimodalWorkspace() {
  authSection.classList.add("hidden");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");
  multimodalSection.classList.remove("hidden");
  const parsed = parseModelSelect(multimodalModelSelect.value);
  activeMultimodalModel = parsed.model;
  activeMultimodalMode = parsed.mode;
  statusDiv.textContent = `Multimodal Preview (${activeMultimodalModel} · ${activeMultimodalMode})`;
  statusDiv.className = "status connected";
}

function resetMultimodalMediaPreview() {
  if (multimodalObjectUrl) {
    URL.revokeObjectURL(multimodalObjectUrl);
    multimodalObjectUrl = null;
  }
  multimodalImagePreview.removeAttribute("src");
  multimodalVideoPreview.removeAttribute("src");
  multimodalImagePreview.classList.add("hidden");
  multimodalVideoPreview.classList.add("hidden");
  multimodalPlaceholder.classList.remove("hidden");
}

function updateMultimodalMediaNote() {
  const parts = [];
  if (multimodalVideoBlob) parts.push("recorded video");
  if (multimodalUploadedFile) parts.push(multimodalUploadedFile.name);
  multimodalMediaNote.textContent = parts.length
    ? `Saved: ${parts.join(" + ")}. Add text if needed, then click Send.`
    : "Text-only prompts can be sent without recording or uploading.";
}

function clearMultimodalDraft() {
  discardMultimodalRecording = true;
  stopMultimodalRecording();
  stopMultimodalRecordingTracks();
  recordBtn.textContent = "Record";
  multimodalSendBtn.disabled = false;
  multimodalTextInput.value = "";
  multimodalFile.value = "";
  multimodalVideoChunks = [];
  multimodalVideoBlob = null;
  multimodalUploadedFile = null;
  resetMultimodalMediaPreview();
  updateMultimodalMediaNote();
}

function resetMultimodalConversation() {
  clearMultimodalDraft();
  multimodalChatLog.innerHTML = "";
  multimodalHistory = [];
  multimodalCameraBtn.textContent = "Start Camera";
  multimodalMicBtn.textContent = "Start Mic";
  multimodalVideoPreview.srcObject = null;
  recordBtn.disabled = true;
  latestMultimodalInputTokens = 0;
  latestMultimodalOutputTokens = 0;
  updateMultimodalTokenDisplay();
}

function stopMultimodalRecordingTracks() {
  if (multimodalCameraStream) {
    multimodalCameraStream.getTracks().forEach(t => t.stop());
    multimodalCameraStream = null;
  }
  if (multimodalMicStream) {
    multimodalMicStream.getTracks().forEach(t => t.stop());
    multimodalMicStream = null;
  }
  multimodalRecordStream = null;
}

function stopMultimodalRecording() {
  if (multimodalRecorder && multimodalRecorder.state !== "inactive") {
    multimodalRecorder.stop();
  }
}

openMultimodalBtn.onclick = openMultimodalWorkspace;

closeMultimodalBtn.onclick = () => {
  stopMultimodalRecording();
  resetUI();
};

recordBtn.onclick = async () => {
  if (multimodalRecorder && multimodalRecorder.state === "recording") {
    stopMultimodalRecording();
    return;
  }

  const tracks = [];
  if (multimodalCameraStream) tracks.push(...multimodalCameraStream.getVideoTracks());
  if (multimodalMicStream) tracks.push(...multimodalMicStream.getAudioTracks());

  if (tracks.length === 0) {
    alert("Start the camera first before recording.");
    return;
  }

  multimodalRecordStream = new MediaStream(tracks);
  multimodalVideoBlob = null;
  multimodalVideoChunks = [];

  multimodalRecorder = new MediaRecorder(multimodalRecordStream, {
    mimeType: "video/webm;codecs=vp8,opus",
  });

  multimodalRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) multimodalVideoChunks.push(event.data);
  };

  multimodalRecorder.onstop = () => {
    if (discardMultimodalRecording) {
      multimodalVideoBlob = null;
      multimodalVideoChunks = [];
    } else {
      multimodalVideoBlob = new Blob(multimodalVideoChunks, { type: "video/webm" });
    }
    discardMultimodalRecording = false;
    recordBtn.textContent = "Record";
    multimodalSendBtn.disabled = false;
    updateMultimodalMediaNote();
  };

  multimodalRecorder.start();
  discardMultimodalRecording = false;
  recordBtn.textContent = "Stop";
  multimodalSendBtn.disabled = true;
  multimodalMediaNote.textContent = "Recording… Click Stop to save.";
};

multimodalFile.addEventListener("change", () => {
  const file = multimodalFile.files && multimodalFile.files[0];
  multimodalUploadedFile = file || null;
  resetMultimodalMediaPreview();

  if (file) {
    multimodalObjectUrl = URL.createObjectURL(file);
    multimodalPlaceholder.classList.add("hidden");

    if (file.type.startsWith("image/")) {
      multimodalImagePreview.src = multimodalObjectUrl;
      multimodalImagePreview.classList.remove("hidden");
    } else if (file.type.startsWith("video/")) {
      multimodalVideoPreview.src = multimodalObjectUrl;
      multimodalVideoPreview.classList.remove("hidden");
    }
  }

  updateMultimodalMediaNote();
});

multimodalModelSelect.addEventListener("change", () => {
  const parsed = parseModelSelect(multimodalModelSelect.value);
  activeMultimodalModel = parsed.model;
  activeMultimodalMode = parsed.mode;
  resetMultimodalConversation();
  statusDiv.textContent = `Multimodal Preview (${activeMultimodalModel} · ${activeMultimodalMode})`;
});

// Connect Button Handler
connectBtn.onclick = async () => {
  if (activeFlow !== "realtime") return;

  const selected = document.querySelector('input[name="provider"]:checked');
  activeProvider = selected ? selected.value : "gemini";

  statusDiv.textContent = `Connecting to ${PROVIDER_LABELS[activeProvider] || activeProvider}...`;
  connectBtn.disabled = true;

  try {
    await mediaHandler.initializeAudio();
    geminiClient.connect(activeProvider);
  } catch (error) {
    console.error("Connection error:", error);
    statusDiv.textContent = "Connection Failed: " + error.message;
    statusDiv.className = "status error";
    connectBtn.disabled = false;
  }
};

multimodalCameraBtn.onclick = async () => {
  if (multimodalCameraStream) {
    multimodalCameraStream.getVideoTracks().forEach(t => t.stop());
    multimodalCameraStream = null;
    multimodalVideoPreview.srcObject = null;
    multimodalVideoPreview.classList.add("hidden");
    multimodalPlaceholder.classList.remove("hidden");
    multimodalCameraBtn.textContent = "Start Camera";
    recordBtn.disabled = true;
    return;
  }

  try {
    multimodalCameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    multimodalVideoPreview.muted = true;
    multimodalVideoPreview.srcObject = multimodalCameraStream;
    multimodalVideoPreview.classList.remove("hidden");
    multimodalPlaceholder.classList.add("hidden");
    multimodalVideoPreview.play().catch(e => console.error("play() failed:", e));
    multimodalCameraBtn.textContent = "Stop Camera";
    recordBtn.disabled = false;
  } catch (e) {
    console.error(e);
    alert("Could not access camera — check permissions");
  }
};

multimodalMicBtn.onclick = async () => {
  if (multimodalMicStream) {
    multimodalMicStream.getAudioTracks().forEach(t => t.stop());
    multimodalMicStream = null;
    multimodalMicBtn.textContent = "Start Mic";
    return;
  }

  try {
    multimodalMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    multimodalMicBtn.textContent = "Stop Mic";
  } catch (e) {
    console.error(e);
    alert("Could not access microphone — check permissions");
  }
};

disconnectBtn.onclick = () => {
  geminiClient.disconnect();
};

micBtn.onclick = async () => {
  if (mediaHandler.isRecording) {
    mediaHandler.stopAudio();
    micBtn.textContent = "Start Mic";
  } else {
    try {
      await mediaHandler.startAudio((data) => {
        if (geminiClient.isConnected()) geminiClient.send(data);
      });
      micBtn.textContent = "Stop Mic";
    } catch (e) {
      alert("Could not start audio capture");
    }
  }
};

cameraBtn.onclick = async () => {
  if (cameraBtn.textContent === "Stop Camera") {
    mediaHandler.stopVideo(videoPreview);
    cameraBtn.textContent = "Start Camera";
    screenBtn.textContent = "Share Screen";
    videoPlaceholder.classList.remove("hidden");
  } else {
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      screenBtn.textContent = "Share Screen";
    }
    try {
      await mediaHandler.startVideo(videoPreview, (base64Data) => {
        if (geminiClient.isConnected()) geminiClient.sendImage(base64Data);
      });
      cameraBtn.textContent = "Stop Camera";
      screenBtn.textContent = "Share Screen";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      alert("Could not access camera");
    }
  }
};

screenBtn.onclick = async () => {
  if (screenBtn.textContent === "Stop Sharing") {
    mediaHandler.stopVideo(videoPreview);
    screenBtn.textContent = "Share Screen";
    cameraBtn.textContent = "Start Camera";
    videoPlaceholder.classList.remove("hidden");
  } else {
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      cameraBtn.textContent = "Start Camera";
    }
    try {
      await mediaHandler.startScreen(
        videoPreview,
        (base64Data) => {
          if (geminiClient.isConnected()) geminiClient.sendImage(base64Data);
        },
        () => {
          screenBtn.textContent = "Share Screen";
          videoPlaceholder.classList.remove("hidden");
        }
      );
      screenBtn.textContent = "Stop Sharing";
      cameraBtn.textContent = "Start Camera";
      videoPlaceholder.classList.add("hidden");
    } catch (e) {
      alert("Could not share screen");
    }
  }
};

sendBtn.onclick = sendText;
textInput.onkeypress = (e) => { if (e.key === "Enter") sendText(); };
multimodalSendBtn.onclick = sendMultimodalPreview;
multimodalTextInput.onkeypress = (e) => { if (e.key === "Enter") sendMultimodalPreview(); };

function sendText() {
  const text = textInput.value;
  if (text && geminiClient.isConnected()) {
    geminiClient.sendText(text);
    appendMessage("user", text);
    textInput.value = "";
  }
}

async function sendMultimodalPreview() {
  const text = multimodalTextInput.value.trim();
  const hasVideo = Boolean(multimodalVideoBlob);
  const hasMedia = Boolean(multimodalUploadedFile);

  if (!text && !hasVideo && !hasMedia) return;

  const requestParts = [];
  if (text) requestParts.push(text);
  if (hasVideo) requestParts.push("[recorded video]");
  if (hasMedia) requestParts.push(`[${multimodalUploadedFile.type || "media"}: ${multimodalUploadedFile.name}]`);

  // ── User bubble with send timestamp ──────────────────────────────────────
  const sentAt = new Date();
  appendMultimodalMessage("user", requestParts.join("\n"), { sentAt });

  // ── Pending model bubble ──────────────────────────────────────────────────
  const { finalize } = appendMultimodalMessage("gemini", "Generating…", { pending: true });

  const formData = new FormData();
  formData.append("model", activeMultimodalModel);
  formData.append("mode", activeMultimodalMode);
  formData.append("prompt", text);
  formData.append("history", JSON.stringify(multimodalHistory));

  if (hasMedia) formData.append("media", multimodalUploadedFile);
  if (hasVideo) {
    const videoFile = new File([multimodalVideoBlob], "recording.webm", { type: "video/webm" });
    formData.append("media", videoFile);
  }

  multimodalSendBtn.disabled = true;
  recordBtn.disabled = true;

  try {
    const response = await fetch("/multimodal", { method: "POST", body: formData });
    const payload = await response.json();

    if (!response.ok) throw new Error(payload.detail || "Multimodal request failed");

    const receivedAt = new Date();
    const assistantText = payload.text || "The model returned an empty response.";

    // ── Finalize model bubble: fill text + latency delta ──────────────────
    finalize(assistantText, receivedAt, sentAt);

    multimodalHistory.push({ role: "user", parts: [{ text: requestParts.join("\n") }] });
    multimodalHistory.push({ role: "model", parts: [{ text: assistantText }] });

    latestMultimodalInputTokens = payload.usage?.input_tokens || 0;
    latestMultimodalOutputTokens = payload.usage?.output_tokens || 0;
    updateMultimodalTokenDisplay();
    clearMultimodalDraft();
  } catch (error) {
    console.error("Multimodal request failed:", error);
    const receivedAt = new Date();
    finalize(`Error: ${error.message}`, receivedAt, sentAt);
    multimodalSendBtn.disabled = false;
  } finally {
    recordBtn.disabled = false;
  }
}

function resetUI() {
  authSection.classList.remove("hidden");
  appSection.classList.add("hidden");
  multimodalSection.classList.add("hidden");
  structuredSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");

  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
  stopMultimodalRecording();
  stopMultimodalRecordingTracks();
  videoPlaceholder.classList.remove("hidden");

  micBtn.textContent = "Start Mic";
  cameraBtn.textContent = "Start Camera";
  screenBtn.textContent = "Share Screen";
  recordBtn.textContent = "Record";
  chatLog.innerHTML = "";
  resetMultimodalConversation();
  const parsedOnReset = parseModelSelect(multimodalModelSelect.value);
  activeMultimodalModel = parsedOnReset.model;
  activeMultimodalMode = parsedOnReset.mode;
  connectBtn.disabled = false;

  multimodalCameraBtn.textContent = "Start Camera";
  multimodalMicBtn.textContent = "Start Mic";
  multimodalVideoPreview.srcObject = null;
  recordBtn.disabled = true;

  latestInputTokens = 0;
  latestOutputTokens = 0;
  updateTokenDisplay();
  updateMultimodalTokenDisplay();
  updateFlowUI();
}

function showSessionEnd() {
  appSection.classList.add("hidden");
  sessionEndSection.classList.remove("hidden");
  mediaHandler.stopAudio();
  mediaHandler.stopVideo(videoPreview);
}

restartBtn.onclick = () => { resetUI(); };

// ── Structured Output: itinerary consistency tester ─────────────────────────
//
// Pure text-in, structured-JSON-out. No media. Each provider's chosen model is
// called `runs` times with identical input; every run is independently
// validated server-side against a deeply nested JSON Schema. Invalid runs show
// the real error — not papered over with a retry or a "fixed" result.

function openStructuredWorkspace() {
  authSection.classList.add("hidden");
  appSection.classList.add("hidden");
  sessionEndSection.classList.add("hidden");
  structuredSection.classList.remove("hidden");
  statusDiv.textContent = "Structured Output";
  statusDiv.className = "status connected";
}

openStructuredBtn.onclick = openStructuredWorkspace;

closeStructuredBtn.onclick = () => {
  structuredSection.classList.add("hidden");
  resetUI();
};

function resetItineraryResults() {
  ["gemini", "qwen"].forEach((provider) => {
    document.getElementById(`${provider}-itinerary-runs`).innerHTML = "";
    document.getElementById(`${provider}-itinerary-summary`).textContent = "";
    document.getElementById(`${provider}-itinerary-summary`).className = "itinerary-summary-badge";
  });
}

structuredClearBtn.onclick = resetItineraryResults;

/**
 * Parse a schema validation error string into its path and message components.
 * Input format: "Schema validation error at root{path}: {message}"
 * Returns { path: string|null, message: string }
 */
function parseValidationError(errorStr) {
  if (!errorStr) return { path: null, message: errorStr || "" };
  const m = errorStr.match(/^Schema validation error at root([^:]*): ([\s\S]+)$/);
  if (!m) return { path: null, message: errorStr };
  return { path: m[1], message: m[2] };
}

/**
 * Navigate a parsed JSON object using a path string like ".days[1].activities[0]".
 * Returns the value at that path, or undefined if not found.
 */
function getAtPath(obj, pathStr) {
  if (!pathStr) return obj;
  let cur = obj;
  const regex = /\.([^.\[]+)|\[(\d+)\]/g;
  let m;
  while ((m = regex.exec(pathStr)) !== null) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = m[1] !== undefined ? cur[m[1]] : cur[parseInt(m[2], 10)];
  }
  return cur;
}

function renderRunCard(run) {
  const card = document.createElement("div");
  card.className = `run-card ${run.valid ? "valid" : "invalid"}`;

  const header = document.createElement("div");
  header.className = "run-card-header";
  const badge = document.createElement("span");
  badge.className = `run-badge ${run.valid ? "valid" : "invalid"}`;
  badge.textContent = run.valid ? "✓ Valid" : "✗ Invalid";
  const meta = document.createElement("span");
  meta.className = "run-meta";
  meta.textContent = `Run ${run.run_index + 1} · ${run.latency_seconds}s`;
  header.appendChild(badge);
  header.appendChild(meta);
  card.appendChild(header);

  if (!run.valid) {
    const errorEl = document.createElement("div");
    errorEl.className = "run-error";
    errorEl.textContent = run.error || "Unknown error";
    card.appendChild(errorEl);

    // If partial_data is present (JSON parsed but schema invalid), show context at error path.
    if (run.partial_data != null) {
      const { path, message } = parseValidationError(run.error);
      if (path !== null) {
        const ctxVal = getAtPath(run.partial_data, path);
        if (ctxVal !== undefined) {
          const ctx = document.createElement("div");
          ctx.className = "run-error-context";
          const label = document.createElement("div");
          label.className = "run-error-context-label";
          label.textContent = `Context at root${path || " (root)"}:`;
          const pre = document.createElement("pre");
          pre.textContent = JSON.stringify(ctxVal, null, 2);
          ctx.appendChild(label);
          ctx.appendChild(pre);
          card.appendChild(ctx);
        }
      }
    }
  }

  if (run.usage) {
    const usageEl = document.createElement("div");
    usageEl.className = "run-usage";
    usageEl.textContent = `In: ${run.usage.input_tokens} · Out: ${run.usage.output_tokens} · Total: ${run.usage.total_tokens}`;
    card.appendChild(usageEl);
  }

  const raw = run.data ? JSON.stringify(run.data, null, 2)
    : run.partial_data ? JSON.stringify(run.partial_data, null, 2)
    : (run.raw_text || "(no output)");
  const details = document.createElement("details");
  details.className = "run-details";
  const summary = document.createElement("summary");
  summary.textContent = run.valid ? "View itinerary JSON" : "View full output";
  const pre = document.createElement("pre");
  pre.textContent = raw;
  details.appendChild(summary);
  details.appendChild(pre);
  card.appendChild(details);

  return card;
}

async function runItineraryProvider(provider, model, destination, days, preference, runs) {
  const runsContainer = document.getElementById(`${provider}-itinerary-runs`);
  const summaryBadge = document.getElementById(`${provider}-itinerary-summary`);
  runsContainer.innerHTML = "";
  summaryBadge.textContent = "Generating…";
  summaryBadge.className = "itinerary-summary-badge loading";

  const fd = new FormData();
  fd.append("model", model);
  fd.append("destination", destination);
  fd.append("days", days);
  fd.append("preference", preference);
  fd.append("runs", runs);

  try {
    const resp = await fetch("/itinerary/structured", { method: "POST", body: fd });
    const payload = await resp.json();
    if (!resp.ok) throw new Error(payload.detail || "Request failed");

    payload.runs.forEach((run) => runsContainer.appendChild(renderRunCard(run)));

    summaryBadge.textContent = `${payload.valid_count}/${payload.total_runs} valid`;
    summaryBadge.className = `itinerary-summary-badge ${payload.valid_count === payload.total_runs ? "all-valid" : "partial"}`;
  } catch (err) {
    console.error(`${provider} itinerary error:`, err);
    summaryBadge.textContent = "Request failed";
    summaryBadge.className = "itinerary-summary-badge invalid";
    const errorCard = document.createElement("div");
    errorCard.className = "run-card invalid";
    errorCard.textContent = `Error: ${err.message}`;
    runsContainer.appendChild(errorCard);
  }
}

structuredSendBtn.onclick = async () => {
  const destination = destinationInput.value.trim();
  const days = parseInt(daysInput.value, 10) || 1;
  const preference = preferenceInput.value.trim();
  const runs = Math.max(1, Math.min(parseInt(runsInput.value, 10) || 3, 5));

  if (!destination || !preference) {
    alert("Fill in destination and preference first.");
    return;
  }

  const geminiModel = structuredGeminiSelect.value;
  const qwenModel = structuredQwenSelect.value;

  structuredSendBtn.disabled = true;

  await Promise.all([
    runItineraryProvider("gemini", geminiModel, destination, days, preference, runs),
    runItineraryProvider("qwen", qwenModel, destination, days, preference, runs),
  ]);

  structuredSendBtn.disabled = false;
};

updateFlowUI();
updateMultimodalTokenDisplay();

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
let multimodalAudioChunks = [];
let multimodalAudioBlob = null;
let multimodalUploadedFile = null;
let multimodalObjectUrl = null;
let activeMultimodalModel = "gemini-3.1-flash-lite";
let activeMultimodalMode = "function_call";
let discardMultimodalRecording = false;

// Add near your other multimodal state vars
let multimodalHistory = []; // { role: "user"|"model", parts: [{ text }] }


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

    // geminiClient.sendText(PROVIDER_INTROS[activeProvider] || PROVIDER_INTROS.gemini);
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
      currentUserMessageDiv.textContent += msg.text;
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentUserMessageDiv = appendMessage("user", msg.text);
    }
  } else if (msg.type === "gemini") {
    if (currentGeminiMessageDiv) {
      currentGeminiMessageDiv.textContent += msg.text;
      chatLog.scrollTop = chatLog.scrollHeight;
    } else {
      currentGeminiMessageDiv = appendMessage("gemini", msg.text);
    }
  }
}

function appendMessage(type, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;
  msgDiv.textContent = text;
  chatLog.appendChild(msgDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
  return msgDiv;
}

function appendMultimodalMessage(type, text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${type}`;
  msgDiv.textContent = text;
  multimodalChatLog.appendChild(msgDiv);
  multimodalChatLog.scrollTop = multimodalChatLog.scrollHeight;
  return msgDiv;
}

function updateFlowUI() {
  const selected = document.querySelector('input[name="flow"]:checked');
  activeFlow = selected ? selected.value : "realtime";

  const isRealtime = activeFlow === "realtime";
  realtimeOptions.classList.toggle("hidden", !isRealtime);
  connectBtn.classList.toggle("hidden", !isRealtime);
  openMultimodalBtn.classList.toggle("hidden", isRealtime);
  statusDiv.textContent = isRealtime
    ? "Disconnected"
    : "Multimodal Preview";
  statusDiv.className = isRealtime ? "status disconnected" : "status connected";
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

  if (multimodalAudioBlob) {
    parts.push("recorded audio");
  }

  if (multimodalUploadedFile) {
    parts.push(multimodalUploadedFile.name);
  }

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
  multimodalAudioChunks = [];
  multimodalAudioBlob = null;
  multimodalUploadedFile = null;
  resetMultimodalMediaPreview();
  updateMultimodalMediaNote();
}

function resetMultimodalConversation() {
  clearMultimodalDraft();
  multimodalChatLog.innerHTML = "";
  multimodalHistory = [];
  latestMultimodalInputTokens = 0;
  latestMultimodalOutputTokens = 0;
  updateMultimodalTokenDisplay();
}

function stopMultimodalRecordingTracks() {
  if (multimodalRecordStream) {
    multimodalRecordStream.getTracks().forEach((track) => track.stop());
    multimodalRecordStream = null;
  }
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

  try {
    multimodalAudioBlob = null;
    multimodalAudioChunks = [];
    multimodalRecordStream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });
    multimodalRecorder = new MediaRecorder(multimodalRecordStream);

    multimodalRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        multimodalAudioChunks.push(event.data);
      }
    };

    multimodalRecorder.onstop = () => {
      if (discardMultimodalRecording) {
        multimodalAudioBlob = null;
        multimodalAudioChunks = [];
      } else {
        multimodalAudioBlob = new Blob(multimodalAudioChunks, {
          type: multimodalRecorder.mimeType || "audio/webm",
        });
      }
      discardMultimodalRecording = false;
      recordBtn.textContent = "Record";
      multimodalSendBtn.disabled = false;
      stopMultimodalRecordingTracks();
      updateMultimodalMediaNote();
    };

    multimodalRecorder.start();
    discardMultimodalRecording = false;
    recordBtn.textContent = "Stop";
    multimodalSendBtn.disabled = true;
    multimodalMediaNote.textContent = "Recording audio. Click Stop to save it before sending.";
  } catch (error) {
    console.error("Could not start multimodal recording:", error);
    alert("Could not start audio recording");
  }
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

  statusDiv.textContent = `Connecting to ${PROVIDER_LABELS[activeProvider] || activeProvider
    }...`;
  connectBtn.disabled = true;

  try {
    // Initialize audio context on user gesture
    await mediaHandler.initializeAudio();

    geminiClient.connect(activeProvider);
  } catch (error) {
    console.error("Connection error:", error);
    statusDiv.textContent = "Connection Failed: " + error.message;
    statusDiv.className = "status error";
    connectBtn.disabled = false;
  }
};

// UI Controls
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
        if (geminiClient.isConnected()) {
          geminiClient.send(data);
        }
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
    // If another stream is active (e.g. Screen), stop it first
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      screenBtn.textContent = "Share Screen";
    }

    try {
      await mediaHandler.startVideo(videoPreview, (base64Data) => {
        if (geminiClient.isConnected()) {
          geminiClient.sendImage(base64Data);
        }
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
    // If another stream is active (e.g. Camera), stop it first
    if (mediaHandler.videoStream) {
      mediaHandler.stopVideo(videoPreview);
      cameraBtn.textContent = "Start Camera";
    }

    try {
      await mediaHandler.startScreen(
        videoPreview,
        (base64Data) => {
          if (geminiClient.isConnected()) {
            geminiClient.sendImage(base64Data);
          }
        },
        () => {
          // onEnded callback (e.g. user stopped sharing from browser)
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
textInput.onkeypress = (e) => {
  if (e.key === "Enter") sendText();
};
multimodalSendBtn.onclick = sendMultimodalPreview;
multimodalTextInput.onkeypress = (e) => {
  if (e.key === "Enter") sendMultimodalPreview();
};

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
  const hasAudio = Boolean(multimodalAudioBlob);
  const hasMedia = Boolean(multimodalUploadedFile);
  console.log(JSON.stringify(multimodalHistory));
  if (!text && !hasAudio && !hasMedia) {
    return;
  }

  const requestParts = [];

  // requestParts.push(`Model: ${activeMultimodalModel}`);

  if (text) {
    requestParts.push(text);
  }

  if (hasAudio) {
    requestParts.push("[recorded audio]");
  }

  if (hasMedia) {
    requestParts.push(`[${multimodalUploadedFile.type || "media"}: ${multimodalUploadedFile.name}]`);
  }

  appendMultimodalMessage("user", requestParts.join("\n"));
  const pendingMessage = appendMultimodalMessage("gemini", "Generating...");

  const formData = new FormData();
  formData.append("model", activeMultimodalModel);
  formData.append("mode", activeMultimodalMode);
  formData.append("prompt", text);
  formData.append("history", JSON.stringify(multimodalHistory)); // ← new


  if (hasMedia) {
    formData.append("media", multimodalUploadedFile);
  }

  if (hasAudio) {
    const audioFile = new File([multimodalAudioBlob], "recording.webm", {
      type: multimodalAudioBlob.type || "audio/webm",
    });
    formData.append("audio", audioFile);
  }

  multimodalSendBtn.disabled = true;
  recordBtn.disabled = true;


  try {
    const response = await fetch("/multimodal", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Multimodal request failed");
    }

    const assistantText = payload.text || "The model returned an empty response.";
    pendingMessage.textContent = assistantText;

    // Push both turns into history after success
    multimodalHistory.push({ role: "user", parts: [{ text: requestParts.join("\n") }] });
    multimodalHistory.push({ role: "model", parts: [{ text: assistantText }] });


    latestMultimodalInputTokens = payload.usage?.input_tokens || 0;
    latestMultimodalOutputTokens = payload.usage?.output_tokens || 0;
    updateMultimodalTokenDisplay();
    clearMultimodalDraft();
  } catch (error) {
    console.error("Multimodal request failed:", error);
    pendingMessage.textContent = `Error: ${error.message}`;
    multimodalSendBtn.disabled = false;
  } finally {
    recordBtn.disabled = false;
  }
}

function resetUI() {
  authSection.classList.remove("hidden");
  appSection.classList.add("hidden");
  multimodalSection.classList.add("hidden");
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

restartBtn.onclick = () => {
  resetUI();
};

updateFlowUI();
updateMultimodalTokenDisplay();

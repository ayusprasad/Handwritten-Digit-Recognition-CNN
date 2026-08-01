const canvas = document.querySelector("#drawCanvas");
const ctx = canvas.getContext("2d");
const fileInput = document.querySelector("#fileInput");
const predictBtn = document.querySelector("#predictBtn");
const clearBtn = document.querySelector("#clearBtn");
const previewName = document.querySelector("#previewName");
const resultDigit = document.querySelector("#resultDigit");
const confidence = document.querySelector("#confidence");
const modelUsed = document.querySelector("#modelUsed");
const bars = document.querySelector("#bars");
const segments = [...document.querySelectorAll(".segment")];
const metricDataset = document.querySelector("#metricDataset");
const metricAccuracy = document.querySelector("#metricAccuracy");

let drawing = false;
let uploadedFile = null;
let dataset = "auto";

function resetCanvas() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 24;
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDrawing(event) {
  drawing = true;
  uploadedFile = null;
  fileInput.value = "";
  previewName.textContent = "Drawing input ready.";
  const point = pointerPosition(event);
  ctx.beginPath();
  ctx.moveTo(point.x, point.y);
}

function draw(event) {
  if (!drawing) return;
  const point = pointerPosition(event);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();
}

function stopDrawing() {
  drawing = false;
}

function canvasBlob() {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

function isCanvasBlank() {
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] < 245 || data[i + 1] < 245 || data[i + 2] < 245) {
      return false;
    }
  }
  return true;
}

function displayLabel(item) {
  return String(item.digit);
}

function renderBars(probabilities) {
  bars.innerHTML = "";
  probabilities.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>${displayLabel(item)}</span>
      <span class="track"><span class="fill" style="width:${item.confidence * 100}%"></span></span>
      <span>${Math.round(item.confidence * 100)}%</span>
    `;
    bars.appendChild(row);
  });
}

function renderMetrics() {
  const metrics = window.MODEL_METRICS?.[dataset];
  metricDataset.textContent = dataset;
  metricAccuracy.textContent =
    typeof metrics?.test_accuracy === "number"
      ? `${Math.round(metrics.test_accuracy * 10000) / 100}%`
      : dataset === "auto"
        ? "best match"
        : "--";
}

async function predict() {
  if (!uploadedFile && isCanvasBlank()) {
    confidence.textContent = "Draw a digit or upload an image first.";
    return;
  }

  const form = new FormData();
  form.append("dataset", dataset);
  form.append("image", uploadedFile || await canvasBlob(), "digit.png");

  confidence.textContent = "Predicting...";
  predictBtn.disabled = true;

  try {
    const response = await fetch("/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");
    resultDigit.textContent = displayLabel(data);
    confidence.textContent = `${Math.round(data.confidence * 10000) / 100}% confidence`;
    modelUsed.textContent = `Model: ${data.selected_dataset || data.dataset || dataset}`;
    renderBars(data.probabilities);
  } catch (error) {
    confidence.textContent = error.message;
  } finally {
    predictBtn.disabled = false;
  }
}

segments.forEach((button) => {
  button.addEventListener("click", () => {
    dataset = button.dataset.dataset;
    segments.forEach((segment) => segment.classList.toggle("active", segment === button));
    renderMetrics();
  });
});

fileInput.addEventListener("change", () => {
  uploadedFile = fileInput.files[0] || null;
  previewName.textContent = uploadedFile ? uploadedFile.name : "Draw a digit or upload an image.";
});

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", () => {
  uploadedFile = null;
  fileInput.value = "";
  resetCanvas();
  resultDigit.textContent = "--";
  confidence.textContent = "Waiting for input";
  modelUsed.textContent = "Model: --";
  bars.innerHTML = "";
  previewName.textContent = "Draw a digit or upload an image.";
});

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
window.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);

resetCanvas();
renderMetrics();
modelUsed.textContent = "Model: --";

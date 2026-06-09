const canvas = document.querySelector("#drawCanvas");
const ctx = canvas.getContext("2d");
const fileInput = document.querySelector("#fileInput");
const predictBtn = document.querySelector("#predictBtn");
const clearBtn = document.querySelector("#clearBtn");
const previewName = document.querySelector("#previewName");
const resultDigit = document.querySelector("#resultDigit");
const confidence = document.querySelector("#confidence");
const bars = document.querySelector("#bars");
const segments = [...document.querySelectorAll(".segment")];
const metricDataset = document.querySelector("#metricDataset");
const metricAccuracy = document.querySelector("#metricAccuracy");

let drawing = false;
let uploadedFile = null;
let dataset = "devanagari";

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
  metricAccuracy.textContent = metrics?.test_accuracy
    ? `${Math.round(metrics.test_accuracy * 10000) / 100}%`
    : "--";
}

async function predict() {
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
  bars.innerHTML = "";
  previewName.textContent = "Draw a digit or upload an image.";
});

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
window.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);

resetCanvas();
renderMetrics();

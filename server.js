const tf = require("@tensorflow/tfjs");
const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const { createCanvas, loadImage } = require("canvas");

const IMAGE_SIZE = 128;
const PORT = 8080;

const app = express();
app.use(cors());
const upload = multer({ dest: "uploads/" });

let model;

// Serve model directory statically
app.use("/model", express.static(path.join(__dirname, "model")));

// Add ping route
app.get("/ping", (req, res) => {
  res.json({ status: "ok", message: "Server is running" });
});

/**
 * Convert uploaded image to a Tensor suitable for model input
 */
async function loadImageAsTensor(imgPath) {
  const img = await loadImage(imgPath);
  const canvas = createCanvas(IMAGE_SIZE, IMAGE_SIZE);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  return tf.browser.fromPixels(canvas)
    .toFloat()
    .div(tf.scalar(255.0))
    .reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
}

/**
 * Run a test prediction on startup to verify model + pipeline
 */
async function runStartupTest() {
  // const imagePath = "./test/test-INCORRECT.jpeg"; // Replace with any local image path
  const imagePath = "./test/test.jpeg"; // Replace with any local image path
  if (!fs.existsSync(imagePath)) {
    console.warn("⚠️ Test image not found. Skipping startup test.");
    return;
  }

  const tensor = await loadImageAsTensor(imagePath);
  const prediction = model.predict(tensor);
  const score = (await prediction.data())[0];
  tensor.dispose();
  prediction.dispose();

  console.log(`📷 Startup Test Prediction: ${score.toFixed(4)}`);
  console.log(score >= 0.5 ? "✅ Classified as: CORRECT" : "❌ Classified as: INCORRECT");
}

/**
 * Load model and start the server
 */
async function startServer() {
  app.listen(PORT, async () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    try {
      model = await tf.loadLayersModel(`http://localhost:${PORT}/model/model.json`);
      console.log("✅ Model loaded successfully.");
      await runStartupTest();
    } catch (err) {
      console.error("❌ Failed to load model:", err);
    }
  });
}

/**
 * Handle prediction requests
 */
app.post("/predict", upload.single("image"), async (req, res) => {
  console.log(`📩 Incoming prediction request from ${req.ip}`);
  if (!req.file) return res.status(400).send("No file uploaded.");

  try {
    const tensor = await loadImageAsTensor(req.file.path);
    const prediction = model.predict(tensor);
    const score = (await prediction.data())[0];
    tensor.dispose();
    prediction.dispose();
    fs.unlinkSync(req.file.path); // Clean up temp file

    const result = {
      score: score.toFixed(4),
      classification: score >= 0.5 ? "CORRECT" : "INCORRECT"
    };

    console.log(`📤 Responding with: ${JSON.stringify(result)}`);
    res.json(result);
  } catch (err) {
    console.error("❌ Prediction error:", err);
    res.status(500).send("Prediction failed.");
  }
});

startServer();

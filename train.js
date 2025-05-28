const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const IMAGE_SIZE = 128;

// Load images from a folder
async function loadImagesFromDir(dirPath, label) {
  const files = fs.readdirSync(dirPath);
  const images = [];

  console.log(`📂 Loading images from ${dirPath}...`);

  for (const file of files) {
    const imgPath = path.join(dirPath, file);
    try {
      const img = await loadImage(imgPath);
      const canvas = createCanvas(IMAGE_SIZE, IMAGE_SIZE);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
      const imageTensor = tf.browser.fromPixels(canvas)
        .toFloat()
        .div(tf.scalar(255.0))
        .reshape([IMAGE_SIZE, IMAGE_SIZE, 3]);
      images.push({ tensor: imageTensor, label });
      console.log(`✅ Loaded ${imgPath}`);
    } catch (err) {
      console.error(`❌ Failed to load ${imgPath}: ${err.message}`);
    }
  }

  console.log(`✅ Finished loading ${images.length} images from ${dirPath}`);
  return images;
}

async function loadDataset() {
  console.log('🧪 Loading dataset...');
  const correct = await loadImagesFromDir('./data/correct', 1);
  const incorrect = await loadImagesFromDir('./data/incorrect', 0);
  const all = correct.concat(incorrect);
  console.log(`📊 Total samples: ${all.length}`);

  const xs = tf.stack(all.map(i => i.tensor));
  const ys = tf.tensor(all.map(i => i.label)).reshape([all.length, 1]);

  console.log('✅ Dataset tensors prepared.');
  return { xs, ys };
}

function createModel() {
  console.log('⚙️ Creating model...');
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
    filters: 16,
    kernelSize: 3,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    loss: 'binaryCrossentropy',
    optimizer: tf.train.adam(),
    metrics: ['accuracy']
  });

  console.log('✅ Model ready.');
  return model;
}

(async () => {
  const { xs, ys } = await loadDataset();
  const model = createModel();

  console.log('🚀 Starting training...');
  await model.fit(xs, ys, {
    epochs: 2,
    batchSize: 8,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`📈 Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} accuracy=${logs.acc.toFixed(4)}`);
      }
    }
  });

  // Manual model saving since 'file://' is not supported in tfjs (browser version)
  const saveResult = await model.save(tf.io.withSaveHandler(async (data) => {
    const modelJson = JSON.stringify({
      modelTopology: data.modelTopology,
      format: 'layers-model',
      generatedBy: 'TensorFlow.js',
      convertedBy: null
    });

    const weightData = data.weightData;

    fs.mkdirSync('./model', { recursive: true });
    fs.writeFileSync('./model/model.json', modelJson);
    fs.writeFileSync('./model/weights.bin', Buffer.from(weightData));

    console.log('✅ Model saved manually to ./model');
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON', weightDataBytes: weightData.byteLength } };
  }));
})();

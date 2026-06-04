const tf = require("@tensorflow/tfjs-node");
const mnist = require("mnist");

// 1. Prepare the Data
console.log("Loading and preparing MNIST data...");
// Fetch 60,000 training images and 10,000 test images
const mnistData = mnist.set(60000, 10000);

function convertToTensors(dataSubset) {
  const images = [];
  const labels = [];

  for (const sample of dataSubset) {
    images.push(sample.input); // Flat array of 784 normalized pixels (0-1)
    labels.push(sample.output); // One-hot encoded array of 10 classes
  }

  // Convert arrays into 2D Tensors
  const xs = tf.tensor2d(images, [images.length, 784]);
  const ys = tf.tensor2d(labels, [labels.length, 10]);
  return { xs, ys };
}

const trainData = convertToTensors(mnistData.training);
const testData = convertToTensors(mnistData.test);

// 2. Build a Simple Sequential Model
console.log("Building the model architecture...");
const model = tf.sequential();

// Input hidden layer: 784 inputs -> 128 hidden units with ReLU activation
model.add(
  tf.layers.dense({
    inputShape: [784],
    units: 128,
    activation: "relu",
  }),
);

// Output layer: 10 units with Softmax for classification probabilities
model.add(
  tf.layers.dense({
    units: 10,
    activation: "softmax",
  }),
);

// 3. Compile the Model
model.compile({
  optimizer: tf.train.adam(),
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

// 4. Train the Model
async function runTraining() {
  console.log("Starting training process...");

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 5,
    batchSize: 64,
    validationData: [testData.xs, testData.ys],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${logs.acc.toFixed(4)}, Val Accuracy = ${logs.val_acc.toFixed(4)}`);
      },
    },
  });

  console.log("\nTraining complete!");

  // 5. Evaluate the Model
  console.log("Evaluating test dataset...");
  const evalResult = model.evaluate(testData.xs, testData.ys);
  const testLoss = evalResult[0].dataSync()[0];
  const testAcc = evalResult[1].dataSync()[0];

  console.log(`Final Test Loss: ${testLoss.toFixed(4)}`);
  console.log(`Final Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);

  // Clean up tensors from memory
  trainData.xs.dispose();
  trainData.ys.dispose();
  testData.xs.dispose();
  testData.ys.dispose();
}

runTraining();

import * as tf from "@tensorflow/tfjs-node";
import mnist, { MnistSample } from "mnist";
import { ReferenceBitNetLayer } from "./lib/ReferenceBitNetLayer";

interface TensorDataPayload {
  xs: tf.Tensor2D;
  ys: tf.Tensor2D;
}

console.log("Loading and preparing MNIST data...");
// Fetch 60,000 training images and 10,000 test images safely typed
const mnistData = mnist.set(60000, 10000);

function convertToTensors(dataSubset: MnistSample[]): TensorDataPayload {
  const images: number[][] = [];
  const labels: number[][] = [];

  for (const sample of dataSubset) {
    images.push(sample.input);
    labels.push(sample.output);
  }

  // Explicitly convert arrays into typed 2D Tensors
  const xs: tf.Tensor2D = tf.tensor2d(images, [images.length, 784]);
  const ys: tf.Tensor2D = tf.tensor2d(labels, [labels.length, 10]);
  return { xs, ys };
}

const trainData: TensorDataPayload = convertToTensors(mnistData.training);
const testData: TensorDataPayload = convertToTensors(mnistData.test);

// 2. Build a Simple Sequential Model
console.log("Building the model architecture...");
const model: tf.Sequential = tf.sequential();

// Input hidden layer: 784 inputs -> 128 hidden units with ReLU activation
model.add(
  new ReferenceBitNetLayer({
    inputShape: [784],
    units: 128,
    activation: "relu",
  }),
);

// Output layer: 10 units with Softmax for classification probabilities
model.add(
  new ReferenceBitNetLayer({
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
async function runTraining(): Promise<void> {
  console.log("Starting training process...");

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 5,
    batchSize: 64,
    validationData: [testData.xs, testData.ys],
    callbacks: {
      onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
        if (logs) {
          // Fallbacks added because keys can vary slightly depending on the environment configuration
          const loss = logs.loss ?? 0;
          const acc = logs.acc ?? logs.accuracy ?? 0;
          const valAcc = logs.val_acc ?? logs.val_accuracy ?? 0;

          console.log(`Epoch ${epoch + 1}: Loss = ${loss.toFixed(4)}, Accuracy = ${acc.toFixed(4)}, Val Accuracy = ${valAcc.toFixed(4)}`);
        }
      },
    },
  });

  console.log("\nTraining complete!");

  // 5. Evaluate the Model
  console.log("Evaluating test dataset...");

  // FIX: model.evaluate can return a single Tensor or an array of Tensors.
  // We force cast it to an array here to safely unpack it in TypeScript.
  const evalResult = model.evaluate(testData.xs, testData.ys) as tf.Scalar[];

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

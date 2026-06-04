import * as tf from "@tensorflow/tfjs-node";
import * as mnist from "mnist";
import { BitNetLayer } from "./lib/BitNetLayer";
import { BitNetOptimizer } from "./lib/BitNetOptimizer";
import { UncompromisedQatAdamStrategy } from "./lib/strategies/UncompromisedQatAdamStrategy";

// Define local interfaces to cleanly map the untyped 'mnist' module data structures
interface MnistSample {
  input: number[];
  output: number[];
}

interface MnistDataset {
  xs: tf.Tensor2D;
  ys: tf.Tensor2D;
}

// 1. Prepare the Data
console.log("Loading and preparing MNIST data...");

// Fetch 60,000 training images and 10,000 test images
const mnistData = (mnist as any).set(60000, 10000);

function convertToTensors(dataSubset: MnistSample[]): MnistDataset {
  const images: number[][] = [];
  const labels: number[][] = [];

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

// 2. Build our BitNet Model Architecture
console.log("Building the model architecture...");

// Instantiate your strategies
const strategyHidden = new UncompromisedQatAdamStrategy(128, 784);
const strategyOutput = new UncompromisedQatAdamStrategy(10, 128);

// Instantiate matching optimizers for each strategy layer
const hiddenOptimizer = new BitNetOptimizer(strategyHidden, 0.001);
const outputOptimizer = new BitNetOptimizer(strategyOutput, 0.001);

const model = tf.sequential();

// Hidden Layer: 784 inputs -> 128 hidden units with ReLU activation
const hiddenLayer = new BitNetLayer({
  inputShape: [784],
  units: 128,
  strategy: strategyHidden,
});
model.add(hiddenLayer);
model.add(tf.layers.activation({ activation: "relu" }));

// Output layer: 10 units with Softmax for classification probabilities
const outputLayer = new BitNetLayer({
  units: 10,
  strategy: strategyOutput,
});
model.add(outputLayer);
model.add(tf.layers.activation({ activation: "softmax" }));

// 3. Compile the Model
model.compile({
  optimizer: "sgd", // Dummy string to satisfy fit requirements
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
      onBatchEnd: async (batch, logs) => {
        tf.tidy(() => {
          // Calculate the specific mini-batch tensor slices from our dataset
          const currentBatchSize = 64;
          const startIdx = batch * currentBatchSize;

          const xBatch = trainData.xs.slice([startIdx, 0], [currentBatchSize, 784]);
          const yBatch = trainData.ys.slice([startIdx, 0], [currentBatchSize, 10]);

          // Manually track gradients on our custom forward pass step
          const costGrads = tf.variableGrads(() => {
            const predictions = model.predict(xBatch) as tf.Tensor;
            return tf.losses.softmaxCrossEntropy(yBatch, predictions) as tf.Scalar;
          });

          // Fetch your unique, compiled layer kernel names
          const hiddenName = hiddenLayer.kernelName;
          const outputName = outputLayer.kernelName;

          // Route the calculated gradient arrays down to the specific layers
          if (costGrads.grads[hiddenName]) {
            hiddenOptimizer.applyGradients({ [hiddenName]: costGrads.grads[hiddenName] });
          }
          if (costGrads.grads[outputName]) {
            outputOptimizer.applyGradients({ [outputName]: costGrads.grads[outputName] });
          }
        });
      },
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: ` + `Loss = ${logs?.loss.toFixed(4)}, ` + `Accuracy = ${logs?.acc.toFixed(4)}, ` + `Val Accuracy = ${logs?.val_acc.toFixed(4)}`);
      },
    },
  });

  console.log("\nTraining complete!");

  // 5. Evaluate the Model
  console.log("Evaluating test dataset...");
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

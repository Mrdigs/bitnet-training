import * as tf from "@tensorflow/tfjs-node";
import mnist, { MnistSample } from "mnist";
import { ReferenceBitNetStrategy } from "./lib/strategy/ReferenceBitNetStrategy";
import { StrategyBitNetLayer } from "./lib/StrategyBitNetLayer";
import { StrategyBitNetOptimizer } from "./lib/StrategyBitNetOptimizer";
import { LearningRate } from "./lib/LearningRate";
import { TrainingMonitorCallback } from "./lib/callback/TrainingMonitorCallback";
import { StochasticBitNetStrategy } from "./lib/strategy/StochasticBitNetStrategy";
import { TerminalChartCallback } from "./lib/callback/TerminalChartCallback";
import { FirstPassLossCallback } from "./lib/callback/FirstPassLossCallback";
import { TernaryWeightDistributionCallback } from "./lib/callback/TernaryWeightDistributionCallback";
import { UnOptimisedRefStrategy } from "./lib/strategy/UnOptimisedRefStrategy";
import { MiniBitNetStrategy } from "./lib/strategy/MiniBitNetStrategy";
import { MagnitudeRefStrategy } from "./lib/strategy/MagnitudeRefStrategy";
import { MagMiniBitNetStrategy } from "./lib/strategy/MagMiniBitNetStrategy";
import { AdamMiniBitNetStrategy } from "./lib/strategy/AdamMiniBitNetStrategy";
import { SpectralRefStrategy } from "./lib/strategy/SpectralRefStrategy";
import { SparseMiniBitNetStrategy } from "./lib/strategy/SparseMiniBitNetStrategy";

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

// NOTE: THIS WAS ACTUALLY A VERY GOOD RESULT!!
// const strategy = new StochasticBitNetStrategy(60.0, 0.5);

//const strategy = new StochasticBitNetStrategy();
//const strategy = new StochasticBitNetStrategy(40.0, 0.5);
//const strategy = new ReferenceBitNetStrategy();
//const strategy = new UnOptimisedRefStrategy();
//const strategy = new SpectralRefStrategy();
// const strategy = new MagnitudeRefStrategy();
//const strategy = new AdamMiniBitNetStrategy();
// const strategy = new MiniBitNetStrategy();
//const strategy = new MagMiniBitNetStrategy();
const strategy = new SparseMiniBitNetStrategy();

const learningRate = new LearningRate((step: number) => {
  /*
  const START_LR = 0.05; // Adjust based on your current baseline
  const END_LR = 0.001;
  const TOTAL_STEPS = 4690; // 5 epochs * ~938 steps per epoch
  const progress = Math.min(step / TOTAL_STEPS, 1.0);
  const currentLR = START_LR - progress * (START_LR - END_LR);
  return currentLR;
  */
  return 0.05;
  // INTERESTING: SEE, ITS THE LEARNING RATE THAT MASSIVELY SCLAES
  // THE GRADIENTS DOWN - OBVIOUSLY. HOW ABOUT TAKING ANOTHER APPROACH?
  // return 1;
});
const optimizer = new StrategyBitNetOptimizer(strategy, learningRate);

const callbacks = [
  // Reports loss on first pass
  // new FirstPassLossCallback(),
  // General training monitoring
  new TrainingMonitorCallback(),
  // Prints a loss curve
  new TerminalChartCallback(),
];

// Input hidden layer: 784 inputs -> 128 hidden units with ReLU activation
model.add(
  new StrategyBitNetLayer({
    strategy,
    learningRate,
    inputShape: [784],
    units: 128,
    activation: "relu",
  }),
);

// Output layer: 10 units with Softmax for classification probabilities
model.add(
  new StrategyBitNetLayer({
    strategy,
    learningRate,
    units: 10,
    activation: "softmax",
  }),
);

/*
callbacks.push(
  // Reports weight distributions
  // @ts-expect-error fuck you
  new TernaryWeightDistributionCallback(model, model.layers[0].name),
);
*/

// 3. Compile the Model
model.compile({
  optimizer,
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
    callbacks,
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

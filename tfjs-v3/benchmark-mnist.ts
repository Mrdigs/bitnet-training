import mnist from "mnist";
import * as asciichart from "asciichart";
import * as tf from "@tensorflow/tfjs-node";
import { BitNetLayer } from "./lib/BitNetLayer";
import { BitNetOptimizer } from "./lib/BitNetOptimizer";
import { IBitNetStrategy } from "./lib/IBitNetStrategy";

// 1. Structural strategy factory closure blueprint typing
export type StrategyFactory = (units: number, inFeatures: number) => IBitNetStrategy;

// 2. Strict candidate parameter interface block grouping
export interface BenchmarkCandidate {
  label: string;
  factory: StrategyFactory;
  learningRate: number;
}

interface MnistSample {
  input: number[];
  output: number[];
}

interface MnistSet {
  training: MnistSample[];
  test: MnistSample[];
}

interface TrainingHistory {
  lossHistory: number[];
  accuracyHistory: number[];
}

/**
 * Builds a 2-layer sequential BitNet model with identical weight snapshots.
 * Leverages custom weights array initialization for strict baseline replication.
 */
function buildIsolatedModel(factory: StrategyFactory, hiddenUnits: number, outputUnits: number, wHiddenInit: tf.Tensor2D, wOutputInit: tf.Tensor2D): tf.LayersModel {
  const model = tf.sequential();

  // Hidden Layer
  model.add(
    new BitNetLayer({
      units: hiddenUnits,
      inputShape: [784],
      strategy: factory(hiddenUnits, 784),
      weights: [wHiddenInit],
    }),
  );
  model.add(tf.layers.activation({ activation: "relu" }));

  // Classification Output Head
  model.add(
    new BitNetLayer({
      units: outputUnits,
      strategy: factory(outputUnits, hiddenUnits),
      weights: [wOutputInit],
    }),
  );
  model.add(tf.layers.activation({ activation: "softmax" }));

  model.compile({
    optimizer: "sgd", // Dummy string to satisfy fit() pipeline requirements
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

/**
 * Executes a custom, mini-batch training loop to inject custom strategy updates.
 */
function trainCandidate(model: tf.LayersModel, factory: StrategyFactory, images: Float32Array, labels: Uint8Array, batchSize: number, learningRate: number, samplesCount: number): TrainingHistory {
  const history: TrainingHistory = { lossHistory: [], accuracyHistory: [] };
  const inputDim = 784;

  // Spin up two independent optimizer nodes tracking the layers' respective layout configurations
  const hiddenLayerStrategy = factory(128, 784);
  const outputLayerStrategy = factory(10, 128);
  const bitNetOptimizer = new BitNetOptimizer(hiddenLayerStrategy, learningRate);

  for (let startIdx = 0; startIdx < samplesCount; startIdx += batchSize) {
    const endIdx = Math.min(startIdx + batchSize, samplesCount);
    const currentBatchSize = endIdx - startIdx;

    tf.tidy(() => {
      // Slice raw typed dataset values straight into localized tensor blocks
      const imgSlice = images.subarray(startIdx * inputDim, endIdx * inputDim);
      const lblSlice = labels.subarray(startIdx, endIdx);

      const xBatch = tf.tensor2d(imgSlice, [currentBatchSize, inputDim], "float32");
      const yBatch = tf.oneHot(tf.tensor1d(lblSlice, "int32"), 10).toFloat();

      // Explicitly capture loss gradients with respect to structural parameters
      const costGrads = tf.variableGrads(() => {
        const predictions = model.predict(xBatch) as tf.Tensor;
        return tf.losses.softmaxCrossEntropy(yBatch, predictions) as tf.Scalar;
      });

      // Record telemetry evaluations
      history.lossHistory.push(costGrads.value.dataSync()[0]);

      const preds = (model.predict(xBatch) as tf.Tensor).argMax(-1);
      const targets = yBatch.argMax(-1);
      const correct = tf.equal(preds, targets).sum().dataSync()[0];
      history.accuracyHistory.push(correct / currentBatchSize);

      // Mutate unmanaged variable buffers using custom strategy registers
      bitNetOptimizer.applyGradients(costGrads.grads);
    });
  }

  return history;
}

/**
 * Universal evaluation harness comparing two strategy configurations side-by-side.
 */
export function runEvaluationHarness(candidateA: BenchmarkCandidate, candidateB: BenchmarkCandidate): void {
  console.log(`=== 🚀 INITIALIZING BITNET MNIST BENCHMARK HARNESS ===`);
  console.log(`Comparing [A]: ${candidateA.label} vs [B]: ${candidateB.label}\n`);

  // 1. Pull data arrays from the 'mnist' ecosystem with explicit cast
  const samplesCount = 1000; // Scaled to 1000 for a snappy micro-benchmark verification run
  const mnistData = (mnist as any).set(samplesCount, 10) as MnistSet;

  const trainImages = new Float32Array(mnistData.training.reduce((acc: number[], d: MnistSample) => acc.concat(d.input), []));
  const trainLabels = new Uint8Array(mnistData.training.map((d: MnistSample) => d.output.indexOf(1)));

  // 2. Generate a fixed random seed weight matrix baseline
  const hiddenUnits = 128;
  const outputUnits = 10;

  const wHiddenInit = tf.randomNormal([hiddenUnits, 784], 0.0, 0.05, "float32", 42) as tf.Tensor2D;
  const wOutputInit = tf.randomNormal([outputUnits, hiddenUnits], 0.0, 0.05, "float32", 42) as tf.Tensor2D;

  // =========================================================================
  // RUN CANDIDATE A
  // =========================================================================
  console.log(`-> Booting Candidate A: ${candidateA.label}...`);
  const modelA = buildIsolatedModel(candidateA.factory, hiddenUnits, outputUnits, wHiddenInit, wOutputInit);
  const resultsA = trainCandidate(modelA, candidateA.factory, trainImages, trainLabels, 64, candidateA.learningRate, samplesCount);
  console.log(`   Final Evaluation Loss State: ${resultsA.lossHistory[resultsA.lossHistory.length - 1].toFixed(5)}`);

  // =========================================================================
  // RUN CANDIDATE B
  // =========================================================================
  console.log(`\n-> Booting Candidate B: ${candidateB.label}...`);
  const modelB = buildIsolatedModel(candidateB.factory, hiddenUnits, outputUnits, wHiddenInit, wOutputInit);
  const resultsB = trainCandidate(modelB, candidateB.factory, trainImages, trainLabels, 64, candidateB.learningRate, samplesCount);
  console.log(`   Final Evaluation Loss State: ${resultsB.lossHistory[resultsB.lossHistory.length - 1].toFixed(5)}`);

  // =========================================================================
  // TERMINAL VISUALIZATION VIA ASCIICHART
  // =========================================================================
  console.log(`\n=== 📊 COMPARATIVE LOSS CONVERGENCE PROGRESSION ===`);
  console.log(`   (Blue/Upper: ${candidateA.label} | Red/Lower: ${candidateB.label})`);

  const stride = 2;
  const plotPointsA: number[] = [];
  const plotPointsB: number[] = [];

  for (let i = 0; i < resultsA.lossHistory.length; i += stride) {
    plotPointsA.push(resultsA.lossHistory[i]);
    plotPointsB.push(resultsB.lossHistory[i]);
  }

  console.log(
    asciichart.plot([plotPointsA, plotPointsB], {
      height: 12,
      colors: [asciichart.blue, asciichart.red],
    }),
  );

  // Explicit unmanaged resource memory dump
  wHiddenInit.dispose();
  wOutputInit.dispose();
}

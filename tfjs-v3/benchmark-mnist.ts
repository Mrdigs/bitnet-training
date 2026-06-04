import * as mnist from "mnist";
import * as asciichart from "asciichart";
import * as tf from "@tensorflow/tfjs-node";
import { BitNetLayer } from "./lib/BitNetLayer";
import { BitNetOptimizer } from "./lib/BitNetOptimizer";
import { IBitNetStrategy } from "./lib/IBitNetStrategy";

export type StrategyFactory = (units: number, inFeatures: number) => IBitNetStrategy;

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

function buildIsolatedModel(factory: StrategyFactory, hiddenUnits: number, outputUnits: number, wHiddenInit: tf.Tensor2D, wOutputInit: tf.Tensor2D): tf.LayersModel {
  const model = tf.sequential();

  model.add(
    new BitNetLayer({
      units: hiddenUnits,
      inputShape: [784],
      strategy: factory(hiddenUnits, 784),
      weights: [wHiddenInit],
    }),
  );
  model.add(tf.layers.activation({ activation: "relu" }));

  model.add(
    new BitNetLayer({
      units: outputUnits,
      strategy: factory(outputUnits, hiddenUnits),
      weights: [wOutputInit],
    }),
  );
  model.add(tf.layers.activation({ activation: "softmax" }));

  model.compile({
    optimizer: "sgd",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

function trainCandidate(model: tf.LayersModel, factory: StrategyFactory, images: Float32Array, labels: Uint8Array, batchSize: number, learningRate: number, samplesCount: number): TrainingHistory {
  const history: TrainingHistory = { lossHistory: [], accuracyHistory: [] };
  const inputDim = 784;

  const hiddenLayerStrategy = factory(128, 784);
  const outputLayerStrategy = factory(10, 128);

  const hiddenLayerOptimizer = new BitNetOptimizer(hiddenLayerStrategy, learningRate);
  const outputLayerOptimizer = new BitNetOptimizer(outputLayerStrategy, learningRate);

  const bitNetLayers = model.layers.filter((l) => l instanceof BitNetLayer) as BitNetLayer[];

  // --- SAFE NAMING LOOKUP ---
  // We extract the true registered string names using our public kernelName getter!
  const hiddenKernelName = bitNetLayers[0].kernelName;
  const outputKernelName = bitNetLayers[1].kernelName;

  console.log(`   [Debug Naming] Target Hidden Variable Name: "${hiddenKernelName}"`);
  console.log(`   [Debug Naming] Target Output Variable Name: "${outputKernelName}"`);

  let stepCounter = 0;

  for (let startIdx = 0; startIdx < samplesCount; startIdx += batchSize) {
    const endIdx = Math.min(startIdx + batchSize, samplesCount);
    const currentBatchSize = endIdx - startIdx;
    stepCounter++;

    tf.tidy(() => {
      const imgSlice = images.subarray(startIdx * inputDim, endIdx * inputDim);
      const lblSlice = labels.subarray(startIdx, endIdx);

      const xBatch = tf.tensor2d(imgSlice, [currentBatchSize, inputDim], "float32");
      const yBatch = tf.oneHot(tf.tensor1d(lblSlice, "int32"), 10).toFloat();

      const costGrads = tf.variableGrads(() => {
        const predictions = model.predict(xBatch) as tf.Tensor;
        return tf.losses.softmaxCrossEntropy(yBatch, predictions) as tf.Scalar;
      });

      const currentLoss = costGrads.value.dataSync()[0];
      history.lossHistory.push(currentLoss);

      const preds = (model.predict(xBatch) as tf.Tensor).argMax(-1);
      const targets = yBatch.argMax(-1);
      const currentAcc = tf.equal(preds, targets).sum().dataSync()[0] / currentBatchSize;
      history.accuracyHistory.push(currentAcc);

      // Extract gradients using our verified public layout names
      const hiddenGrad = costGrads.grads[hiddenKernelName];
      const outputGrad = costGrads.grads[outputKernelName];

      if (hiddenGrad) {
        hiddenLayerOptimizer.applyGradients({ [hiddenKernelName]: hiddenGrad });
      }

      if (outputGrad) {
        outputLayerOptimizer.applyGradients({ [outputKernelName]: outputGrad });
      }

      // Live Telemetry Output Frame
      if (stepCounter % 4 === 0 || endIdx === samplesCount) {
        console.log(`   [Batch Step ${String(stepCounter).padStart(2, "0")}] ` + `Processed: ${String(endIdx).padStart(4, "0")}/${samplesCount} | ` + `Loss: ${currentLoss.toFixed(5)} | ` + `Acc: ${(currentAcc * 100).toFixed(2)}%`);
      }
    });
  }

  return history;
}

export function runEvaluationHarness(candidateA: BenchmarkCandidate, candidateB: BenchmarkCandidate): void {
  console.log(`=== 🚀 INITIALIZING BITNET MNIST BENCHMARK HARNESS ===`);
  console.log(`Comparing [A]: ${candidateA.label} vs [B]: ${candidateB.label}\n`);

  const samplesCount = 1000;
  const mnistData = (mnist as any).set(samplesCount, 10) as MnistSet;

  // 1. Extract raw continuous arrays from the package dataset
  const rawImages = mnistData.training.reduce((acc: number[], d: MnistSample) => acc.concat(d.input), [] as number[]);
  const rawLabels = mnistData.training.map((d: MnistSample) => d.output.indexOf(1));

  // 2. Create a randomized shuffling index map array
  const indices = Array.from({ length: samplesCount }, (_, i) => i);

  // High-velocity Fisher-Yates shuffle algorithm with a localized deterministic seed
  // to ensure Candidate A and Candidate B receive an IDENTICAL shuffled mapping order
  let seed = 12345;
  const randomSeeded = () => {
    const x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
  };

  for (let i = samplesCount - 1; i > 0; i--) {
    const j = Math.floor(randomSeeded() * (i + 1));
    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  // 3. Build uniformly balanced shuffled typed buffers
  const trainImages = new Float32Array(samplesCount * 784);
  const trainLabels = new Uint8Array(samplesCount);

  for (let i = 0; i < samplesCount; i++) {
    const originalIndex = indices[i];

    // Copy the 784 feature pixel layout into place
    const srcOffset = originalIndex * 784;
    const destOffset = i * 784;
    for (let p = 0; p < 784; p++) {
      trainImages[destOffset + p] = rawImages[srcOffset + p];
    }

    // Copy the scalar class target label index into place
    trainLabels[i] = rawLabels[originalIndex];
  }

  // 4. Generate a fixed random seed weight matrix baseline
  const hiddenUnits = 128;
  const outputUnits = 10;

  // Reset standard deviation back to standard 0.05 scaling bounds now that shuffling is active
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

  const stride = 1;
  const plotPointsA: number[] = [];
  const plotPointsB: number[] = [];

  for (let i = 0; i < resultsA.lossHistory.length; i += stride) {
    plotPointsA.push(resultsA.lossHistory[i]);
    plotPointsB.push(resultsB.lossHistory[i]);
  }

  const chartConfig: any = {
    height: 14,
    colors: [asciichart.blue, asciichart.red],
  };

  console.log(asciichart.plot([plotPointsA, plotPointsB], chartConfig));

  wHiddenInit.dispose();
  wOutputInit.dispose();
}

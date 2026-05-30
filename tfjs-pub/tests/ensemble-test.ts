import * as tf from "@tensorflow/tfjs";
import { FusedBitNetLayer } from "./layers/fusedBitNet/FusedBitNetLayer";
import { BaselineAdamBitNetLayer } from "../layers/BaselineAdamBitNetLayer";

const NUM_TRIALS = 10; // Run 10 independent trials to establish statistical significance
const BATCH_SIZE = 128;
const TOTAL_STEPS = 400;
const INITIAL_LR = 0.005;
const GRAD_SCALE = 1.5;
const K = 0.5;
const NUM_FEATURES = 64;

async function runEnsembleBenchmark() {
  console.log("=========================================================");
  console.log(`LAUNCHING MONTE CARLO ENSEMBLE: ${NUM_TRIALS} INDEPENDENT TRIALS`);
  console.log("=========================================================");

  const experimentalFinalLosses: number[] = [];
  const baselineFinalLosses: number[] = [];

  for (let trial = 1; trial <= NUM_TRIALS; trial++) {
    console.log(`Executing Trial [${trial}/${NUM_TRIALS}]...`);

    // FIX: Instantiate the layers outside a tidy block. We manage their lifecycles manually.
    const experimentalLayer = new FusedBitNetLayer({ inFeatures: NUM_FEATURES, outFeatures: 1, gradScale: GRAD_SCALE, K });
    const baselineLayer = new BaselineAdamBitNetLayer({ inFeatures: NUM_FEATURES, outFeatures: 1 });

    // Initialize baseline with random variance
    tf.tidy(() => {
      const scale = Math.sqrt(2.0 / (NUM_FEATURES + 1));
      const randomWeights = tf.randomNormal([NUM_FEATURES, 1], 0.0, scale, "float32");
      const oldWeights = (baselineLayer as any).shadowWeights;
      (baselineLayer as any).shadowWeights = tf.keep(randomWeights);
      oldWeights.dispose();
    });

    // The step loop is purely synchronous, allowing tf.tidy to act flawlessly per step
    for (let step = 1; step <= TOTAL_STEPS; step++) {
      tf.tidy(() => {
        const currentLrNormalized = Math.max(0.0, 1.0 - step / TOTAL_STEPS);
        const activeLr = INITIAL_LR * currentLrNormalized;

        const rawX = tf.randomUniform([BATCH_SIZE, 1], -1.0, 1.0, "float32");
        const targetY = tf.add(tf.sin(tf.mul(rawX, tf.scalar(2.0 * Math.PI))), tf.mul(tf.sin(tf.mul(rawX, tf.scalar(6.0 * Math.PI))), tf.scalar(0.5)));

        const featureProjMatrix = tf.sin(tf.mul(rawX, tf.linspace(1, NUM_FEATURES, NUM_FEATURES).reshape([1, NUM_FEATURES])));
        const denseInputs = tf.cast(featureProjMatrix, "float32") as tf.Tensor2D;

        // Forward Pass
        const expPreds = experimentalLayer.forward(denseInputs);
        const basePreds = baselineLayer.forward(denseInputs);

        // Backward Math
        const expLossGradients = tf.div(tf.mul(tf.sub(expPreds, targetY), tf.scalar(2.0)), tf.scalar(BATCH_SIZE, "float32"));
        const expParamGradients = tf.matMul(denseInputs.transpose(), expLossGradients) as tf.Tensor2D;

        const baseLossGradients = tf.div(tf.mul(tf.sub(basePreds, targetY), tf.scalar(2.0)), tf.scalar(BATCH_SIZE, "float32"));
        const baseParamGradients = tf.matMul(denseInputs.transpose(), baseLossGradients) as tf.Tensor2D;

        const clippedExpParamGradients = tf.clipByValue(expParamGradients, -0.1, 0.1) as tf.Tensor2D;
        const clippedBaseParamGradients = tf.clipByValue(baseParamGradients, -0.1, 0.1) as tf.Tensor2D;

        experimentalLayer.applyStep(clippedExpParamGradients, currentLrNormalized);
        baselineLayer.applyStep(clippedBaseParamGradients, activeLr * 100.0);

        // On the absolute final step, extract the single scalar loss back to the host array
        if (step === TOTAL_STEPS) {
          const finalExpLoss = tf.mean(tf.square(tf.sub(expPreds, targetY)));
          const finalBaseLoss = tf.mean(tf.square(tf.sub(basePreds, targetY)));

          // Pull the raw first scalar index out of the Float32Array
          experimentalFinalLosses.push(finalExpLoss.dataSync()[0]);
          baselineFinalLosses.push(finalBaseLoss.dataSync()[0]);
        }
      });
    }

    // Clean up current layer parameters completely from VRAM before initializing the next seed
    experimentalLayer.dispose();
    baselineLayer.dispose();

    // Safely release the main JS thread execution block between intensive trial runs
    await tf.nextFrame();
  }

  // --- COMPUTING MACROSCOPIC ENSEMBLE METRICS ---
  const calcStats = (losses: number[]) => {
    const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
    const variance = losses.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / losses.length;
    return { mean, stdDev: Math.sqrt(variance) };
  };

  const expStats = calcStats(experimentalFinalLosses);
  const baseStats = calcStats(baselineFinalLosses);

  console.log("\n=========================================================");
  console.log("FINAL MONTE CARLO AGGREGATION REPORT                     ");
  console.log("=========================================================");
  console.log(`1-BYTE EXPERIMENTAL STATE MACHINE LAYER (1 Byte State Footprint):`);
  console.log(`  |- Mean Final MSE Loss (μ) : ${expStats.mean.toFixed(5)}`);
  console.log(`  |- Standard Deviation  (σ) : ${expStats.stdDev.toFixed(5)}`);
  console.log(`\n12-BYTE SHADOW WEIGHT ADAM BASELINE LAYER (12 Byte State Footprint):`);
  console.log(`  |- Mean Final MSE Loss (μ) : ${baseStats.mean.toFixed(5)}`);
  console.log(`  |- Standard Deviation  (σ) : ${baseStats.stdDev.toFixed(5)}`);
  console.log("=========================================================");
}

runEnsembleBenchmark();

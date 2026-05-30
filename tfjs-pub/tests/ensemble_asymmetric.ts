import * as tf from "@tensorflow/tfjs";
//import { FusedBitNetLayer } from "./FusedBitNetLayer";
import { FusedBitNetLayer } from "../layers/modularFusedBitNet/FusedBitNetLayer";
import { BaselineAdamBitNetLayer } from "../layers/BaselineAdamBitNetLayer";
import { ProportionalFrictionStrategy } from "../layers/modularFusedBitNet/lib/inertiaStrategy/ProportionalFrictionStrategy";
import { KineticInertiaStrategy } from "../layers/modularFusedBitNet/lib/inertiaStrategy/KineticInertiaStrategy";
import { DirectionAwareLUTStrategy } from "../layers/modularFusedBitNet/lib/gateStrategy/DirectionAwareLUTStrategy";

const NUM_TRIALS = 10;
const BATCH_SIZE = 128;
const INITIAL_LR = 0.005;
const GRAD_SCALE = 1.5;
const K = 0.5;
const NUM_FEATURES = 64;

// --- ASYMMETRIC TIME ALLOCATION ---
const BASELINE_TOTAL_STEPS = 400; // Adam gets a tight 400-step runway
const EXPERIMENTAL_TOTAL_STEPS = 400; // 1-Byte machine gets 4x the steps to convert gradient energy

async function runAsymmetricEnsembleBenchmark() {
  console.log("=========================================================");
  console.log(`LAUNCHING MONTE CARLO ASYMMETRIC TIME TRIAL: ${NUM_TRIALS} INDEPENDENT TRIALS`);
  console.log(`  |- Baseline Budget     : ${BASELINE_TOTAL_STEPS} Steps`);
  console.log(`  |- Experimental Budget : ${EXPERIMENTAL_TOTAL_STEPS} Steps`);
  console.log("=========================================================");

  const experimentalFinalLosses: number[] = [];
  const baselineFinalLosses: number[] = [];

  for (let trial = 1; trial <= NUM_TRIALS; trial++) {
    console.log(`Executing Trial [${trial}/${NUM_TRIALS}]...`);

    const experimentalLayer = new FusedBitNetLayer({ inFeatures: NUM_FEATURES, outFeatures: 1, gradScale: GRAD_SCALE, K });
    const baselineLayer = new BaselineAdamBitNetLayer({ inFeatures: NUM_FEATURES, outFeatures: 1 });

    // Initialize baseline with random variance to break symmetry
    tf.tidy(() => {
      const scale = Math.sqrt(2.0 / (NUM_FEATURES + 1));
      const randomWeights = tf.randomNormal([NUM_FEATURES, 1], 0.0, scale, "float32");
      const oldWeights = (baselineLayer as any).shadowWeights;
      (baselineLayer as any).shadowWeights = tf.keep(randomWeights);
      oldWeights.dispose();
    });

    // Helper to generate fresh wave data per step in the loops
    const generateData = () => {
      const rawX = tf.randomUniform([BATCH_SIZE, 1], -1.0, 1.0, "float32");
      const targetY = tf.add(tf.sin(tf.mul(rawX, tf.scalar(2.0 * Math.PI))), tf.mul(tf.sin(tf.mul(rawX, tf.scalar(6.0 * Math.PI))), tf.scalar(0.5)));
      const featureProjMatrix = tf.sin(tf.mul(rawX, tf.linspace(1, NUM_FEATURES, NUM_FEATURES).reshape([1, NUM_FEATURES])));
      const denseInputs = tf.cast(featureProjMatrix, "float32") as tf.Tensor2D;
      return { denseInputs, targetY };
    };

    // --- PHASE A: BASELINE RUN (400 STEPS) ---
    for (let step = 1; step <= BASELINE_TOTAL_STEPS; step++) {
      tf.tidy(() => {
        const currentLrNormalized = Math.max(0.0, 1.0 - step / BASELINE_TOTAL_STEPS);
        const activeLr = INITIAL_LR * currentLrNormalized;
        const { denseInputs, targetY } = generateData();

        const basePreds = baselineLayer.forward(denseInputs);
        const baseError = tf.sub(basePreds, targetY);
        const baseLossGradients = tf.div(tf.mul(baseError, tf.scalar(2.0)), tf.scalar(BATCH_SIZE, "float32"));
        const baseParamGradients = tf.matMul(denseInputs.transpose(), baseLossGradients) as tf.Tensor2D;
        const clippedBaseParamGradients = tf.clipByValue(baseParamGradients, -0.1, 0.1) as tf.Tensor2D;

        baselineLayer.applyStep(clippedBaseParamGradients, activeLr * 100.0);

        if (step === BASELINE_TOTAL_STEPS) {
          const finalBaseLoss = tf.mean(tf.square(tf.sub(basePreds, targetY)));
          baselineFinalLosses.push(finalBaseLoss.dataSync()[0]);
        }
      });
    }

    // --- PHASE B: EXPERIMENTAL RUN (1,600 STEPS) ---
    for (let step = 1; step <= EXPERIMENTAL_TOTAL_STEPS; step++) {
      tf.tidy(() => {
        // Learning rate decay is smoothly stretched across the full 1,600 steps
        const currentLrNormalized = Math.max(0.0, 1.0 - step / EXPERIMENTAL_TOTAL_STEPS);
        const { denseInputs, targetY } = generateData();

        const expPreds = experimentalLayer.forward(denseInputs);
        const expError = tf.sub(expPreds, targetY);
        const expLossGradients = tf.div(tf.mul(expError, tf.scalar(2.0)), tf.scalar(BATCH_SIZE, "float32"));
        const expParamGradients = tf.matMul(denseInputs.transpose(), expLossGradients) as tf.Tensor2D;
        const clippedExpParamGradients = tf.clipByValue(expParamGradients, -0.1, 0.1) as tf.Tensor2D;

        experimentalLayer.applyStep(clippedExpParamGradients, currentLrNormalized);

        if (step === EXPERIMENTAL_TOTAL_STEPS) {
          const finalExpLoss = tf.mean(tf.square(tf.sub(expPreds, targetY)));
          experimentalFinalLosses.push(finalExpLoss.dataSync()[0]);
        }
      });
    }

    // Clean parameters out of VRAM explicitly
    experimentalLayer.dispose();
    baselineLayer.dispose();

    await tf.nextFrame();
  }

  // --- COMPUTING AGGREGATED METRICS ---
  const calcStats = (losses: number[]) => {
    const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
    const variance = losses.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / losses.length;
    return { mean, stdDev: Math.sqrt(variance) };
  };

  const expStats = calcStats(experimentalFinalLosses);
  const baseStats = calcStats(baselineFinalLosses);

  console.log("\n=========================================================");
  console.log("FINAL ASYMMETRIC TIME TRIAL REPORT                       ");
  console.log("=========================================================");
  console.log(`1-BYTE EXPERIMENTAL STATE MACHINE (1 Byte Footprint @ 1600 Steps):`);
  console.log(`  |- Mean Final MSE Loss (μ) : ${expStats.mean.toFixed(5)}`);
  console.log(`  |- Standard Deviation  (σ) : ${expStats.stdDev.toFixed(5)}`);
  console.log(`\n12-BYTE SHADOW WEIGHT ADAM BASELINE (12 Byte Footprint @ 400 Steps):`);
  console.log(`  |- Mean Final MSE Loss (μ) : ${baseStats.mean.toFixed(5)}`);
  console.log(`  |- Standard Deviation  (σ) : ${baseStats.stdDev.toFixed(5)}`);
  console.log("=========================================================");
}

runAsymmetricEnsembleBenchmark();

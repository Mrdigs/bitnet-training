import * as tf from "@tensorflow/tfjs";
import * as chart from "asciichart";
import { FusedBitNetLayer } from "./layers/modularFusedBitNet/FusedBitNetLayer";
import { BaselineAdamBitNetLayer } from "./layers/BaselineAdamBitNetLayer";
import { ClassicBrakingStrategy } from "./layers/modularFusedBitNet/lib/inertiaStrategy/ClassicBrakingStrategy";
import { KineticInertiaStrategy } from "./layers/modularFusedBitNet/lib/inertiaStrategy/KineticInertiaStrategy";
import { ThresholdAccumulatorLUTStrategy } from "./layers/modularFusedBitNet/lib/gateStrategy/ThresholdAccumulatorLUTStrategy";

const BATCH_SIZE = 128;
const TOTAL_STEPS = 400;
const INITIAL_LR = 0.005;
const GRAD_SCALE = 1.5;
const K = 0.5;
const NUM_FEATURES = 64;

async function runVisualValidation() {
  console.log("=========================================================");
  console.log("LAUNCHING VISUAL BENCHMARK RUN: 400 VS 400 STEP ARENA    ");
  console.log("=========================================================");

  // Instantiate two distinct experimental instances to evaluate our inertia strategies side-by-side
  const classicLayer = new FusedBitNetLayer({
    inFeatures: NUM_FEATURES,
    outFeatures: 1,
    gradScale: GRAD_SCALE,
    K,
    inertiaStrategy: new ClassicBrakingStrategy(),
    gateStrategy: new ThresholdAccumulatorLUTStrategy(),
  });

  const kineticLayer = new FusedBitNetLayer({
    inFeatures: NUM_FEATURES,
    outFeatures: 1,
    gradScale: GRAD_SCALE,
    K,
    inertiaStrategy: new KineticInertiaStrategy(2, 3), // State-aware testing
    gateStrategy: new ThresholdAccumulatorLUTStrategy(),
  });

  const adamLayer = new BaselineAdamBitNetLayer({ inFeatures: NUM_FEATURES, outFeatures: 1 });

  // Symmetry breaking random init for Adam shadow variables
  tf.tidy(() => {
    const scale = Math.sqrt(2.0 / (NUM_FEATURES + 1));
    const randomWeights = tf.randomNormal([NUM_FEATURES, 1], 0.0, scale, "float32");
    const oldW = (adamLayer as any).shadowWeights;
    (adamLayer as any).shadowWeights = tf.keep(randomWeights);
    oldW.dispose();
  });

  // Tracking arrays to capture the entire loss history for chart compilation
  const classicHistory: number[] = [];
  const kineticHistory: number[] = [];
  const adamHistory: number[] = [];

  // Helper to generate fresh continuous wave nodes per step
  const generateData = () => {
    const rawX = tf.randomUniform([BATCH_SIZE, 1], -1.0, 1.0, "float32");
    const targetY = tf.add(tf.sin(tf.mul(rawX, tf.scalar(2.0 * Math.PI))), tf.mul(tf.sin(tf.mul(rawX, tf.scalar(6.0 * Math.PI))), tf.scalar(0.5)));
    const featureProjMatrix = tf.sin(tf.mul(rawX, tf.linspace(1, NUM_FEATURES, NUM_FEATURES).reshape([1, NUM_FEATURES])));
    const denseInputs = tf.cast(featureProjMatrix, "float32") as tf.Tensor2D;
    return { denseInputs, targetY };
  };

  console.log("Grinding training batches entirely inside VRAM. Compiling vectors...");

  for (let step = 1; step <= TOTAL_STEPS; step++) {
    tf.tidy(() => {
      const currentLrNormalized = Math.max(0.0, 1.0 - step / TOTAL_STEPS);
      const activeLr = INITIAL_LR * currentLrNormalized;
      const { denseInputs, targetY } = generateData();

      // --- 1. CLASSIC STRATEGY STEP ---
      const classicPreds = classicLayer.forward(denseInputs);
      const classicLoss = tf.mean(tf.square(tf.sub(classicPreds, targetY)));
      const classicGrads = tf.matMul(denseInputs.transpose(), tf.div(tf.mul(tf.sub(classicPreds, targetY), 2.0), BATCH_SIZE));
      classicLayer.applyStep(tf.clipByValue(classicGrads, -0.1, 0.1) as tf.Tensor2D, currentLrNormalized);

      // --- 2. KINETIC STRATEGY STEP ---
      const kineticPreds = kineticLayer.forward(denseInputs);
      const kineticLoss = tf.mean(tf.square(tf.sub(kineticPreds, targetY)));
      const kineticGrads = tf.matMul(denseInputs.transpose(), tf.div(tf.mul(tf.sub(kineticPreds, targetY), 2.0), BATCH_SIZE));
      kineticLayer.applyStep(tf.clipByValue(kineticGrads, -0.1, 0.1) as tf.Tensor2D, currentLrNormalized);

      // --- 3. FP32 ADAM BASELINE STEP ---
      const adamPreds = adamLayer.forward(denseInputs);
      const adamLoss = tf.mean(tf.square(tf.sub(adamPreds, targetY)));
      const adamGrads = tf.matMul(denseInputs.transpose(), tf.div(tf.mul(tf.sub(adamPreds, targetY), 2.0), BATCH_SIZE));
      adamLayer.applyStep(tf.clipByValue(adamGrads, -0.1, 0.1) as tf.Tensor2D, activeLr * 100.0);

      // Collect step metrics synchronously back to host memory (sampled every step for granular plots)
      classicHistory.push(Number(classicLoss.dataSync()[0]));
      kineticHistory.push(Number(kineticLoss.dataSync()[0]));
      adamHistory.push(Number(adamLoss.dataSync()[0]));
    });

    if (step % 50 === 0) {
      console.log(` -> Step [${step}/${TOTAL_STEPS}] computed successfully...`);
      await tf.nextFrame();
    }
  }

  // --- RENDER VISUAL TEXT CHARTS ---
  console.log("\n=========================================================");
  console.log("VISUAL TRAJECTORY REPORT: THE LOSS EVOLUTION PLOTS       ");
  console.log("=========================================================");

  // We downsample the plotting array to 40 data blocks to keep the ASCII grid readable on screen
  const downsample = (arr: number[], points = 40) => {
    const stepSize = Math.floor(arr.length / points);
    return Array.from({ length: points }, (_, i) => arr[i * stepSize]);
  };

  console.log("\n[1-BYTE STATE MACHINE: CLASSICAL BRAKING STRATEGY LOSS CURVE]");
  console.log(chart.plot(downsample(classicHistory), { height: 10, colors: [chart.cyan] } as any));

  console.log("\n[1-BYTE STATE MACHINE: KINETIC INERTIA STRATEGY LOSS CURVE]");
  console.log(chart.plot(downsample(kineticHistory), { height: 10, colors: [chart.yellow] } as any));

  console.log("\n[12-BYTE SHADOW FLOAT ADAM BASELINE LOSS CURVE]");
  console.log(chart.plot(downsample(adamHistory), { height: 10, colors: [chart.magenta] } as any));

  console.log("=========================================================");
  console.log(`Final Step Results:`);
  console.log(`  |- Classic Strategy (1-Byte) : ${classicHistory[TOTAL_STEPS - 1].toFixed(5)}`);
  console.log(`  |- Kinetic Strategy (1-Byte) : ${kineticHistory[TOTAL_STEPS - 1].toFixed(5)}`);
  console.log(`  |- FP32 Adam Baseline (12-Byte): ${adamHistory[TOTAL_STEPS - 1].toFixed(5)}`);
  console.log("=========================================================");

  classicLayer.dispose();
  kineticLayer.dispose();
  adamLayer.dispose();
}

runVisualValidation();

import * as tf from "@tensorflow/tfjs";
import * as chart from "asciichart";
import { FusedBitNetLayer } from "./layers/modularFusedBitNet/FusedBitNetLayer";
import { BaselineAdamBitNetLayer } from "./layers/BaselineAdamBitNetLayer";
import { DynamicSymmetricalCodec } from "./layers/modularFusedBitNet/lib/codec/DynamicSymmetricalCodec";
import { ProportionalCoolingFlywheelStrategy } from "./layers/modularFusedBitNet/lib/inertiaStrategy/ProportionalCoolingFlywheelStrategy";
import { TernaryStepGateStrategy } from "./layers/modularFusedBitNet/lib/gateStrategy/TernaryStepGateStrategy";
import { SymmetricalVarianceQuantizer } from "./layers/modularFusedBitNet/lib/quantizer/SymmetricalVarianceQuantizer";
import { Fused2bW6bMCodec } from "./layers/modularFusedBitNet/lib/codec/Fused2bW6bMCodec";

// --- PRODUCTION 1-MINUTE HIGH-VOLUME BLUEPRINT ---
const BATCH_SIZE_X = 128; // Standardized continuous batch extraction footprint
const TOTAL_STEPS = 900; // Complete runtime horizon pass
const INITIAL_LR = 0.001;
const GRAD_SCALE = 20.0; // Calibrated kinetic energy scale for real macro manifolds
const K = 0.5;

// High-capacity matrix dimensions layout
const INPUT_DIM = 256;
const OUTPUT_DIM = 64;
const TOTAL_SAMPLES = 4000;

async function runMacroAssetConvergenceArena(baseFriction: number = 1.0): Promise<number> {
  console.log("=========================================================");
  console.log("LAUNCHING DOWNSAMPLED HARMONIZED DATA ARENA (V5.2 SPEC)  ");
  console.log(`Layer Configuration: Wide [${INPUT_DIM} Inputs] -> [${OUTPUT_DIM} Outputs]`);
  console.log("=========================================================\n");

  console.log("Compiling co-integrated 4,000-day macro pipeline asset matrix...");

  // Generate the true underlying hidden physical structural matrix mapping
  const trueHiddenLaw = tf.keep(tf.matMul(tf.randomNormal([INPUT_DIM, 1], 0.0, 0.4, "float32"), tf.randomNormal([1, OUTPUT_DIM], 0.0, 0.4, "float32"))) as tf.Tensor2D;

  // Build the global 4,000-day market history tensor blocks inside VRAM
  const { globalX, globalY } = tf.tidy(() => {
    const rawFeatures = tf.randomUniform([TOTAL_SAMPLES, INPUT_DIM], -2.0, 2.0, "float32");
    const trendLine = tf.sin(tf.linspace(0, 10, TOTAL_SAMPLES)).reshape([TOTAL_SAMPLES, 1]);
    const integratedX = tf.add(rawFeatures, tf.broadcastTo(trendLine, [TOTAL_SAMPLES, INPUT_DIM]));

    // Map features through the true hidden law and inject minor noise variance jitter
    const cleanY = tf.matMul(integratedX, trueHiddenLaw);
    const noiseJitter = tf.randomNormal([TOTAL_SAMPLES, OUTPUT_DIM], 0.0, 0.05, "float32");
    const integratedY = tf.add(cleanY, noiseJitter);

    return { globalX: tf.keep(integratedX), globalY: tf.keep(integratedY) };
  });

  // Generate the master symmetry-breaking random initialization seed matrix
  const scale = Math.sqrt(2.0 / (INPUT_DIM + OUTPUT_DIM));
  const sharedRandomWeights = tf.keep(tf.randomNormal([INPUT_DIM, OUTPUT_DIM], 0.0, scale, "float32")) as tf.Tensor2D;

  // 1. Instantiate the 1-byte production layer instance configuration
  const experimentalLayer = new FusedBitNetLayer({
    inFeatures: INPUT_DIM,
    outFeatures: OUTPUT_DIM,
    initialWeights: sharedRandomWeights,
    gradScale: GRAD_SCALE,
    K,
    codec: new Fused2bW6bMCodec(sharedRandomWeights),
    inertiaStrategy: new ProportionalCoolingFlywheelStrategy(baseFriction, 1.0, 1.5),
  });

  // 2. Instantiate our baseline layer and inject the synchronized starting seed
  const baselineLayer = new BaselineAdamBitNetLayer({
    inFeatures: INPUT_DIM,
    outFeatures: OUTPUT_DIM,
    initialWeights: sharedRandomWeights,
  });

  const expHistory: number[] = [];
  const baseHistory: number[] = [];
  const totalElementsDivisor = BATCH_SIZE_X * OUTPUT_DIM;

  // Generate a held-out test set for evaluating prediction agreement
  const testSize = 500;
  const testX = tf.tidy(() => {
    const rawFeatures = tf.randomUniform([testSize, INPUT_DIM], -2.0, 2.0, "float32");
    const trendLine = tf.sin(tf.linspace(0, 10, testSize)).reshape([testSize, 1]);
    return tf.add(rawFeatures, tf.broadcastTo(trendLine, [testSize, INPUT_DIM])) as tf.Tensor2D;
  });

  // FIX: Core batch generator now strictly extracts live data slices from our pre-allocated market tensors!
  const fetchMarketBatch = () => {
    return tf.tidy(() => {
      const startIdx = Math.floor(Math.random() * (TOTAL_SAMPLES - BATCH_SIZE_X));
      const batchX = (globalX as tf.Tensor2D).slice([startIdx, 0], [BATCH_SIZE_X, INPUT_DIM]);
      const batchY = (globalY as tf.Tensor2D).slice([startIdx, 0], [BATCH_SIZE_X, OUTPUT_DIM]);
      return { batchX, batchY };
    });
  };

  console.log("Beginning parallel optimization pass loops across the market timeline...");

  for (let step = 1; step <= TOTAL_STEPS; step++) {
    tf.tidy(() => {
      const currentLrNormalized = Math.max(0.0, 1.0 - step / TOTAL_STEPS);
      const activeLr = INITIAL_LR * currentLrNormalized;

      // FIX CONSUMPTION PASS: Extract real data tensors from our active market matrix slices
      const { batchX, batchY } = fetchMarketBatch();

      // --- 1. EXPERIMENTAL 1-BYTE STATE MACHINE STEP ---
      const expPreds = experimentalLayer.forward(batchX);
      const expLoss = tf.mean(tf.square(tf.sub(expPreds, batchY)));
      const expGrads = tf.matMul(batchX.transpose(), tf.div(tf.mul(tf.sub(expPreds, batchY), 2.0), tf.scalar(totalElementsDivisor, "float32")));
      //experimentalLayer.applyStep(tf.clipByValue(expGrads, -1.0, 1.0) as tf.Tensor2D, currentLrNormalized);
      experimentalLayer.applyStep(tf.clipByValue(expGrads, -2.5, 2.5) as tf.Tensor2D, currentLrNormalized);

      // --- 2. BASELINE 12-BYTE SHADOW FLOAT ADAM STEP ---
      const basePreds = baselineLayer.forward(batchX);
      const baseLoss = tf.mean(tf.square(tf.sub(basePreds, batchY)));
      const baseGrads = tf.matMul(batchX.transpose(), tf.div(tf.mul(tf.sub(basePreds, batchY), 2.0), tf.scalar(totalElementsDivisor, "float32")));
      //baselineLayer.applyStep(tf.clipByValue(baseGrads, -1.0, 1.0) as tf.Tensor2D, activeLr);
      baselineLayer.applyStep(tf.clipByValue(baseGrads, -2.5, 2.5) as tf.Tensor2D, activeLr);

      expHistory.push(Number(expLoss.dataSync()));
      baseHistory.push(Number(baseLoss.dataSync()));
    });

    if (step % 20 === 0) {
      await tf.nextFrame();
    }
  }

  // --- DISPLAY INTERFACE PLOT AXIS COMPRESSION ---
  const plotPoints = 40;
  const downsample = (arr: number[]) => {
    const stepSize = Math.floor(arr.length / plotPoints);
    return Array.from({ length: plotPoints }, (_, i) => arr[i * stepSize]);
  };

  const plotAndAlignAxis = (history: number[], height: number, color: string) => {
    const compressedData = downsample(history);
    const rawChartStr = chart.plot(compressedData, { height, colors: [color] } as any);
    const chartLines = rawChartStr.split("\n");

    let maxYLabelLength = 0;
    for (let i = 0; i < chartLines.length; i++) {
      const match = chartLines[i].match(/^\s*[\d\.\-]+/);
      if (match && match.length > maxYLabelLength) {
        maxYLabelLength = match.length;
      }
    }

    const pad = " ".repeat(maxYLabelLength + 2);
    const axisLine = pad + "└" + "─".repeat(compressedData.length - 1);
    let labelRow = pad + "0" + " ".repeat(5) + "240" + " ".repeat(5) + "480" + " ".repeat(5) + "720" + " ".repeat(5) + "960" + " ".repeat(3) + "1200 (Steps)";

    console.log(rawChartStr);
    console.log(axisLine);
    console.log(labelRow + "\n");
  };

  console.log("\n=========================================================");
  console.log("PRODUCTION REAL ASSETS CO-INTEGRATION RESULTS            ");
  console.log("=========================================================");
  console.log("[1-BYTE KINETIC MACHINE DEEP CONVERGENCE TIMELINE]");
  plotAndAlignAxis(expHistory, 10, chart.yellow);

  console.log("[12-BYTE FP32 SHADOW ADAM BASELINE CONVERGENCE TIMELINE]");
  plotAndAlignAxis(baseHistory, 10, chart.magenta);
  console.log("=========================================================");
  console.log(`Final Step Results:`);
  console.log(`  |- Fused BitNet Kinetic Layer (1-Byte)     : ${expHistory[TOTAL_STEPS - 1].toFixed(5)}`);
  console.log(`  |- Quantized Adam Baseline Layer (12-Byte) : ${baseHistory[TOTAL_STEPS - 1].toFixed(5)}`);

  // Compare predictions on held-out test set
  const predictionAgreement = tf.tidy(() => {
    const expTestPreds = experimentalLayer.forward(testX);
    const baseTestPreds = baselineLayer.forward(testX);

    // MSE between predictions
    const predDiff = tf.square(tf.sub(expTestPreds, baseTestPreds));
    const mse = tf.mean(predDiff);

    // Correlation: how similar are the outputs?
    const expMean = tf.mean(expTestPreds);
    const baseMean = tf.mean(baseTestPreds);
    const expCentered = tf.sub(expTestPreds, expMean);
    const baseCentered = tf.sub(baseTestPreds, baseMean);

    const covariance = tf.mean(tf.mul(expCentered, baseCentered));
    const expStd = tf.sqrt(tf.mean(tf.square(expCentered)));
    const baseStd = tf.sqrt(tf.mean(tf.square(baseCentered)));
    const correlation = tf.div(covariance, tf.mul(expStd, baseStd));

    return {
      mse: Number(mse.dataSync()[0]),
      correlation: Number(correlation.dataSync()[0]),
    };
  });

  testX.dispose();

  console.log(`\nPrediction Agreement Metrics (Test Set):`);
  console.log(`  |- Prediction MSE (Lower = More Similar)  : ${predictionAgreement.mse.toFixed(6)}`);
  console.log(`  |- Output Correlation (-1 to 1)           : ${predictionAgreement.correlation.toFixed(4)}`);
  console.log("=========================================================");

  experimentalLayer.dispose();
  baselineLayer.dispose();
  sharedRandomWeights.dispose();
  globalX.dispose();
  globalY.dispose();
  trueHiddenLaw.dispose();

  return predictionAgreement.correlation;
}

async function main(baseFrictions: number[] = [0.96]) {
  const correlations: number[] = [];
  for (const friction of baseFrictions) {
    correlations.push(await runMacroAssetConvergenceArena(friction));
  }

  console.log("\n=========================================================");
  console.log("BASE FRICTION SENSITIVITY ANALYSIS SUMMARY              ");
  console.log("=========================================================");
  baseFrictions.forEach((friction, idx) => {
    console.log(`Base Friction: ${friction.toFixed(2)} | Output Correlation: ${correlations[idx].toFixed(4)}`);
  });
}

//main([1.0, 0.98, 0.96, 0.94, 0.92, 0.9]);
//main([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
main([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]);
// main();

import * as tf from "@tensorflow/tfjs";
import * as chart from "asciichart";
import { FusedBitNetLayer } from "./layers/modularFusedBitNet/FusedBitNetLayer";
import { BaselineAdamBitNetLayer } from "./layers/BaselineAdamBitNetLayer";
import { DynamicSymmetricalCodec } from "./layers/modularFusedBitNet/lib/codec/DynamicSymmetricalCodec";
import { ContinuousWarpInertiaStrategy } from "./layers/modularFusedBitNet/lib/inertiaStrategy/ContinuousWarpInertiaStrategy";
import { TernaryStepGateStrategy } from "./layers/modularFusedBitNet/lib/gateStrategy/TernaryStepGateStrategy";
import { SymmetricalVarianceQuantizer } from "./layers/modularFusedBitNet/lib/quantizer/SymmetricalVarianceQuantizer";

// --- TELEMETRY INSPECTION PROFILE ---
const BATCH_SIZE = 64;
const TOTAL_STEPS = 400;
const INITIAL_LR = 0.002;
const GRAD_SCALE = 30.0;
const K = 0.5;
const DIM_SIZE = 128; // Optimized footprint to speed up dataSync host extraction passes

// Pre-allocate a fixed, static target matrix mapping to track true geometric convergence
const targetTrueMatrix = tf.keep(tf.randomNormal([DIM_SIZE, DIM_SIZE], 0.0, 0.5, "float32"));

function generateHistogramBar(count: number, maxCount: number, width: number = 20): string {
  if (maxCount === 0) return "";
  const length = Math.round((count / maxCount) * width);
  return "█".repeat(length);
}

async function runMatrixInspection() {
  console.log("=========================================================");
  console.log(`LAUNCHING INTEGRATED SUB-ATOMIC SYSTEM INSPECTOR         `);
  console.log(`Matrix Grid Capacity: [${DIM_SIZE} x ${DIM_SIZE}] (16,384 Total Registers)`);
  console.log("=========================================================\n");

  const scale = Math.sqrt(2.0 / (DIM_SIZE + DIM_SIZE));
  const sharedRandomWeights = tf.keep(tf.randomNormal([DIM_SIZE, DIM_SIZE], 0.0, scale, "float32")) as tf.Tensor2D;

  const experimentalLayer = new FusedBitNetLayer({
    inFeatures: DIM_SIZE,
    outFeatures: DIM_SIZE,
    gradScale: GRAD_SCALE,
    K,
    codec: new DynamicSymmetricalCodec(sharedRandomWeights),
    inertiaStrategy: new ContinuousWarpInertiaStrategy(0.85),
    gateStrategy: new TernaryStepGateStrategy(),
    quantizer: new SymmetricalVarianceQuantizer(sharedRandomWeights),
  });

  const baselineLayer = new BaselineAdamBitNetLayer({ inFeatures: DIM_SIZE, outFeatures: DIM_SIZE });
  const oldWeights = (baselineLayer as any).shadowWeights;
  (baselineLayer as any).shadowWeights = tf.keep(sharedRandomWeights);
  oldWeights.dispose();

  const expHistory: number[] = [];
  const baseHistory: number[] = [];

  const generateDatasetBatch = () => {
    const rawX = tf.randomUniform([BATCH_SIZE, DIM_SIZE], -1.0, 1.0, "float32");
    const targetY = tf.matMul(rawX, targetTrueMatrix as tf.Tensor2D);
    return { rawX, targetY };
  };

  for (let step = 1; step <= TOTAL_STEPS; step++) {
    tf.tidy(() => {
      const currentLrNormalized = Math.max(0.0, 1.0 - step / TOTAL_STEPS);
      const activeLr = INITIAL_LR * currentLrNormalized;
      const { rawX: rawXData, targetY: targetYData } = generateDatasetBatch();
      const rawX = rawXData as tf.Tensor2D;
      const targetY = targetYData as tf.Tensor2D;

      const expPreds = experimentalLayer.forward(rawX);
      const expLoss = tf.mean(tf.square(tf.sub(expPreds, targetY)));
      const expGrads = tf.matMul(rawX.transpose(), tf.div(tf.mul(tf.sub(expPreds, targetY), 2.0), BATCH_SIZE));
      experimentalLayer.applyStep(tf.clipByValue(expGrads, -2.5, 2.5) as tf.Tensor2D, currentLrNormalized);

      const basePreds = baselineLayer.forward(rawX);
      const baseLoss = tf.mean(tf.square(tf.sub(basePreds, targetY)));
      const baseGrads = tf.matMul(rawX.transpose(), tf.div(tf.mul(tf.sub(basePreds, targetY), 2.0), BATCH_SIZE));
      baselineLayer.applyStep(tf.clipByValue(baseGrads, -2.5, 2.5) as tf.Tensor2D, activeLr * 100.0);

      expHistory.push(Number(expLoss.dataSync()));
      baseHistory.push(Number(baseLoss.dataSync()));
    });

    // --- PHASE TRAJECTORY SAMPLING BLOCK: EVERY 100 STEPS ---
    if (step === 1 || step % 100 === 0 || step === TOTAL_STEPS) {
      console.log(`\n>>> SURGICAL TELEMETRY SNAPSHOT: STEP [${step}/${TOTAL_STEPS}] <<<`);
      console.log(`Current 1-Byte Machine Loss: ${expHistory[step - 1].toFixed(5)}`);
      console.log(`Current FP32 Adam Baseline Loss: ${baseHistory[step - 1].toFixed(5)}`);

      // Extract the live private uint8 array reference synchronously out of VRAM back to host memory
      const rawBuffer = (experimentalLayer as any).packedState.dataSync() as Int32Array;

      let negWeights = 0,
        zeroWeights = 0,
        posWeights = 0;

      // Allocate buckets for momentum bins: Deep Negative, Braking, Cradle Rest, Accelerating, Deep Positive
      let mDeepNeg = 0,
        mNegBrake = 0,
        mCradleRest = 0,
        mPosBrake = 0,
        mDeepPos = 0;

      for (let i = 0; i < rawBuffer.length; i++) {
        const val = rawBuffer[i];

        // Polyfill Token Extraction
        const token = val % 4;
        if (token === 0) negWeights++;
        if (token === 1) zeroWeights++;
        if (token === 2) posWeights++;

        // Polyfill Signed Momentum Extraction
        const rawShifted = Math.floor(val / 4) % 64;
        const mom = rawShifted >= 32 ? rawShifted - 64 : rawShifted;

        if (mom <= -15) mDeepNeg++;
        else if (mom < 0 && mom > -15) mNegBrake++;
        else if (mom === 0) mCradleRest++;
        else if (mom > 0 && mom < 15) mPosBrake++;
        else if (mom >= 15) mDeepPos++;
      }

      const maxWCount = Math.max(negWeights, zeroWeights, posWeights);
      const maxMCount = Math.max(mDeepNeg, mNegBrake, mCradleRest, mPosBrake, mDeepPos);

      console.log("\n  [FORWARD TERNARY WEIGHTS DISTRIBUTION]");
      console.log(`    |- Token 00 (-1.0) : ${negWeights.toString().padEnd(6)} ${generateHistogramBar(negWeights, maxWCount)}`);
      console.log(`    |- Token 01 ( 0.0) : ${zeroWeights.toString().padEnd(6)} ${generateHistogramBar(zeroWeights, maxWCount)}`);
      console.log(`    |- Token 10 (+1.0) : ${posWeights.toString().padEnd(6)} ${generateHistogramBar(posWeights, maxWCount)}`);

      console.log("\n  [6-BIT MOMENTUM ENERGY VELOCITY HISTOGRAM]");
      console.log(`    |- Deep Negative (<= -15)   : ${mDeepNeg.toString().padEnd(6)} ${generateHistogramBar(mDeepNeg, maxMCount)}`);
      console.log(`    |- Negative Drifting (-14..-1) : ${mNegBrake.toString().padEnd(6)} ${generateHistogramBar(mNegBrake, maxMCount)}`);
      console.log(`    |- Rest Cradle Valley (= 0)  : ${mCradleRest.toString().padEnd(6)} ${generateHistogramBar(mCradleRest, maxMCount)}`);
      console.log(`    |- Positive Drifting (1..14) : ${mPosBrake.toString().padEnd(6)} ${generateHistogramBar(mPosBrake, maxMCount)}`);
      console.log(`    |- Deep Positive (>= 15)    : ${mDeepPos.toString().padEnd(6)} ${generateHistogramBar(mDeepPos, maxMCount)}`);
      console.log("─".repeat(57));

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
    let labelRow = pad + "0" + " ".repeat(6) + "100" + " ".repeat(5) + "200" + " ".repeat(5) + "300" + " ".repeat(5) + "400 (Steps)";

    console.log(rawChartStr);
    console.log(axisLine);
    console.log(labelRow + "\n");
  };

  console.log("\n=========================================================");
  console.log("FINAL PERFORMANCE TRAJECTORY COMPARISON                  ");
  console.log("=========================================================");
  console.log("[1-BYTE STATE MACHINE (CONTINUOUS WARP) PURE QAT LOSS]");
  plotAndAlignAxis(expHistory, 10, chart.yellow);

  console.log("[12-BYTE SHADOW FLOAT ADAM BASELINE PURE QAT LOSS]");
  plotAndAlignAxis(baseHistory, 10, chart.magenta);

  experimentalLayer.dispose();
  baselineLayer.dispose();
  targetTrueMatrix.dispose();
  sharedRandomWeights.dispose();
}

runMatrixInspection();

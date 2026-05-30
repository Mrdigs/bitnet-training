import * as tf from "@tensorflow/tfjs";
import * as chart from "asciichart";
import { FusedBitNetLayer } from "./layers/modularFusedBitNet/FusedBitNetLayer";
import { BaselineAdamBitNetLayer } from "./layers/BaselineAdamBitNetLayer";
import { DynamicSymmetricalCodec } from "./layers/modularFusedBitNet/lib/codec/DynamicSymmetricalCodec";
import { ProportionalCoolingFlywheelStrategy } from "./layers/modularFusedBitNet/lib/inertiaStrategy/ProportionalCoolingFlywheelStrategy";
import { TernaryStepGateStrategy } from "./layers/modularFusedBitNet/lib/gateStrategy/TernaryStepGateStrategy";
import { SymmetricalVarianceQuantizer } from "./layers/modularFusedBitNet/lib/quantizer/SymmetricalVarianceQuantizer";

// --- EXPLOSIVE HIGH-LOSS ARENA CONFIGURATION ---
const BATCH_SIZE = 128; // Heavy sample volume for clean discrete macro-aggregates
const TOTAL_STEPS = 500;
const INITIAL_LR = 0.002;
const GRAD_SCALE = 20.0; // Calibrated kinetic energy intake force for high-loss regions
const K = 0.5;
const DIM_SIZE = 256;

// --- FIXED STRUCTURAL LEARNABLE MANIFOLD TARGET ---
// Pre-allocate a high-variance, low-rank outer product transformation matrix.
// Scaling the variance coefficients up forces the initial baseline starting loss
// to land cleanly at an explosive ~11.0 for both models.
const targetTrueMatrix = tf.keep(tf.matMul(tf.randomNormal([DIM_SIZE, 1], 0.0, 0.45, "float32"), tf.randomNormal([1, DIM_SIZE], 0.0, 0.45, "float32"))) as tf.Tensor2D;

async function runHighLossHockeyStickBattle() {
  console.log("=========================================================");
  console.log("LAUNCHING DOWNSAMPLED HIGH-LOSS ARENA (V3.1 PROPORTIONAL)");
  console.log(`Model Layout: High-Capacity [${DIM_SIZE} x ${DIM_SIZE}] Topology`);
  console.log("=========================================================");

  // 1. Generate the master symmetry-breaking random initialization matrix first
  const scale = Math.sqrt(2.0 / (DIM_SIZE + DIM_SIZE));
  const sharedRandomWeights = tf.keep(tf.randomNormal([DIM_SIZE, DIM_SIZE], 0.0, scale, "float32")) as tf.Tensor2D;

  // 2. Instantiate our experimental layer injecting the master random weights natively
  const experimentalLayer = new FusedBitNetLayer({
    inFeatures: DIM_SIZE,
    outFeatures: DIM_SIZE,
    gradScale: GRAD_SCALE,
    K,
    codec: new DynamicSymmetricalCodec(sharedRandomWeights),
    inertiaStrategy: new ProportionalCoolingFlywheelStrategy(0.75, 1.0, 1.5),
    gateStrategy: new TernaryStepGateStrategy(),
    quantizer: new SymmetricalVarianceQuantizer(sharedRandomWeights),
  });

  // 3. Instantiate Adam baseline natively injecting the shared matrix directly through configuration variables
  const baselineLayer = new BaselineAdamBitNetLayer({
    inFeatures: DIM_SIZE,
    outFeatures: DIM_SIZE,
    initialWeights: sharedRandomWeights, // Enforces perfect synchronized initialization scale at birth!
  });

  const expHistory: number[] = [];
  const baseHistory: number[] = [];

  // Dense structural coordinate dataset batch generator
  const generateDatasetBatch = () => {
    const rawX = tf.randomUniform([BATCH_SIZE, DIM_SIZE], -1.0, 1.0, "float32");
    const targetY = tf.matMul(rawX, targetTrueMatrix);
    return { rawX, targetY };
  };

  console.log("Grinding training steps entirely inside VRAM. Compiling vectors...");

  for (let step = 1; step <= TOTAL_STEPS; step++) {
    tf.tidy(() => {
      const currentLrNormalized = Math.max(0.0, 1.0 - step / TOTAL_STEPS);
      const activeLr = INITIAL_LR * currentLrNormalized;
      const { rawX: rawXData, targetY: targetYData } = generateDatasetBatch();

      const rawX = rawXData as tf.Tensor2D;
      const targetY = targetYData as tf.Tensor2D;

      // --- 1. EXPERIMENTAL 1-BYTE STATE MACHINE STEP ---
      const expPreds = experimentalLayer.forward(rawX);
      const expLoss = tf.mean(tf.square(tf.sub(expPreds, targetY)));
      const expGrads = tf.matMul(rawX.transpose(), tf.div(tf.mul(tf.sub(expPreds, targetY), 2.0), BATCH_SIZE));
      experimentalLayer.applyStep(tf.clipByValue(expGrads, -1.0, 1.0) as tf.Tensor2D, currentLrNormalized);

      // --- 2. BASELINE 12-BYTE SHADOW FLOAT ADAM STEP ---
      const basePreds = baselineLayer.forward(rawX);
      const baseLoss = tf.mean(tf.square(tf.sub(basePreds, targetY)));
      const baseGrads = tf.matMul(rawX.transpose(), tf.div(tf.mul(tf.sub(basePreds, targetY), 2.0), BATCH_SIZE));
      baselineLayer.applyStep(tf.clipByValue(baseGrads, -1.0, 1.0) as tf.Tensor2D, activeLr * 100.0);

      expHistory.push(Number(expLoss.dataSync()));
      baseHistory.push(Number(baseLoss.dataSync()));
    });

    if (step % 100 === 0) {
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
    let labelRow = pad + "0" + " ".repeat(6) + "100" + " ".repeat(5) + "200" + " ".repeat(5) + "300" + " ".repeat(5) + "400" + " ".repeat(3) + "500 (Steps)";

    console.log(rawChartStr);
    console.log(axisLine);
    console.log(labelRow + "\n");
  };

  console.log("\n=========================================================");
  console.log("VISUAL TRAJECTORY REPORT: HIGH-LOSS STRUCTURAL ARENA     ");
  console.log("=========================================================");

  console.log("[1-BYTE STATE MACHINE (CONTINUOUS WARP) PURE QAT LOSS CURVE]");
  plotAndAlignAxis(expHistory, 10, chart.yellow);

  console.log("[12-BYTE SHADOW FLOAT ADAM BASELINE PURE QAT LOSS CURVE]");
  plotAndAlignAxis(baseHistory, 10, chart.magenta);

  console.log("=========================================================");
  console.log(`Final Step Results:`);
  console.log(`  |- Fused BitNet Kinetic Layer (1-Byte)     : ${expHistory[TOTAL_STEPS - 1].toFixed(5)}`);
  console.log(`  |- Quantized Adam Baseline Layer (12-Byte) : ${baseHistory[TOTAL_STEPS - 1].toFixed(5)}`);
  console.log("=========================================================");

  experimentalLayer.dispose();
  baselineLayer.dispose();
  targetTrueMatrix.dispose();
  sharedRandomWeights.dispose();
}

runHighLossHockeyStickBattle();

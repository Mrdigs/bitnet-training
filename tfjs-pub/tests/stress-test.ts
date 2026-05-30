import * as tf from "@tensorflow/tfjs";
import { FusedBitNetLayer } from "./layers/fusedBitNet/FusedBitNetLayer";
import { BaselineAdamBitNetLayer } from "../layers/BaselineAdamBitNetLayer";

// --- STRESS TEST CALIBRATION BLUEPRINT ---
const BATCH_SIZE = 128;
const TOTAL_STEPS = 400;
const INITIAL_LR = 0.005; // Learning rate for continuous curve navigation

// FIX: Calibrate intake scale to prevent the massive MSE gradient magnitudes
// from violently overrunning the fixed 6-bit momentum registers.
const GRAD_SCALE = 1.5;
const K = 0.5;

async function runSymmetryStressTest() {
  console.log("=========================================================");
  console.log("LAUNCHING ACID-TEST: MULTI-FREQUENCY WAVE REGRESSION    ");
  console.log("=========================================================");

  // Use a dense hidden mapping size of 64x64 to approximate wave complexity
  const numFeatures = 64;

  const experimentalLayer = new FusedBitNetLayer({ inFeatures: numFeatures, outFeatures: 1, gradScale: GRAD_SCALE, K });
  const baselineLayer = new BaselineAdamBitNetLayer({ inFeatures: numFeatures, outFeatures: 1 });

  // Initialize Adam with random continuous variance to break mathematical symmetry
  tf.tidy(() => {
    const scale = Math.sqrt(2.0 / (numFeatures + 1));
    const randomWeights = tf.randomNormal([numFeatures, 1], 0.0, scale, "float32");
    const oldWeights = (baselineLayer as any).shadowWeights;
    (baselineLayer as any).shadowWeights = tf.keep(randomWeights);
    oldWeights.dispose();
  });

  console.log(`Model Geometry Configured: [${numFeatures} continuous features -> 1 output scalar]`);
  console.log("Running regression on complex compound wave function natively...\n");

  for (let step = 1; step <= TOTAL_STEPS; step++) {
    tf.tidy(() => {
      const currentLrNormalized = Math.max(0.0, 1.0 - step / TOTAL_STEPS);
      const activeLr = INITIAL_LR * currentLrNormalized;

      // 1. Generate DENSE, continuous data points completely in VRAM
      const rawX = tf.randomUniform([BATCH_SIZE, 1], -1.0, 1.0, "float32");

      // Multi-frequency continuous wave target: sin(2pi*x) + 0.5*sin(6pi*x)
      const targetY = tf.add(tf.sin(tf.mul(rawX, tf.scalar(2.0 * Math.PI))), tf.mul(tf.sin(tf.mul(rawX, tf.scalar(6.0 * Math.PI))), tf.scalar(0.5)));

      // Project inputs into a dense, overlapping [BATCH_SIZE, numFeatures] input matrix
      const featureProjMatrix = tf.sin(tf.mul(rawX, tf.linspace(1, numFeatures, numFeatures).reshape([1, numFeatures])));
      const denseInputs = tf.cast(featureProjMatrix, "float32") as tf.Tensor2D;

      // --- FORWARD PASSES (COMPUTING DENSE REGRESSION) ---
      const expPreds = experimentalLayer.forward(denseInputs);
      const expLoss = tf.mean(tf.square(tf.sub(expPreds, targetY)));

      const basePreds = baselineLayer.forward(denseInputs);
      const baseLoss = tf.mean(tf.square(tf.sub(basePreds, targetY)));

      // --- PURE SEPARATED ANALYTICAL MSE GRADIENTS ---
      // Derivative of MSE loss: 2 * (Predictions - Targets) / N
      const expError = tf.sub(expPreds, targetY);
      const expLossGradients = tf.div(tf.mul(expError, tf.scalar(2.0)), tf.scalar(BATCH_SIZE, "float32"));
      const expParamGradients = tf.matMul(denseInputs.transpose(), expLossGradients) as tf.Tensor2D;

      const baseError = tf.sub(basePreds, targetY);
      const baseLossGradients = tf.div(tf.mul(baseError, tf.scalar(2.0)), tf.scalar(BATCH_SIZE, "float32"));
      const baseParamGradients = tf.matMul(denseInputs.transpose(), baseLossGradients) as tf.Tensor2D;

      // --- EXTERNAL LOOP-LEVEL GRADIENT GOVERNOR ---
      // Standard framework practice: Clip the parameter gradients out-of-band before update intake.
      // This prevents massive MSE residual spikes from causing parameter destabilization.
      const clippedExpParamGradients = tf.clipByValue(expParamGradients, -0.1, 0.1) as tf.Tensor2D;
      const clippedBaseParamGradients = tf.clipByValue(baseParamGradients, -0.1, 0.1) as tf.Tensor2D;

      // --- SYNCHRONIZED BACKWARD PATH WRITE-BACK ---
      experimentalLayer.applyStep(clippedExpParamGradients, currentLrNormalized);
      baselineLayer.applyStep(clippedBaseParamGradients, activeLr * 100.0);

      // --- TELEMETRY DASHBOARD ---
      if (step === 1 || step % 50 === 0) {
        const expStates = (experimentalLayer as any).packedState.dataSync() as Int32Array;
        const baseShadow = (baselineLayer as any).shadowWeights.dataSync() as Float32Array;

        let countMinusOne = 0,
          countZero = 0,
          countPlusOne = 0;
        for (let i = 0; i < expStates.length; i++) {
          const wToken = expStates[i] & 0x03;
          if (wToken === 0) countMinusOne++;
          if (wToken === 1) countZero++;
          if (wToken === 2) countPlusOne++;
        }

        let alignmentMatches = 0;
        for (let i = 0; i < baseShadow.length; i++) {
          const val = baseShadow[i];
          const quantizedBaseW = val < -0.5 ? 0 : val < 0.5 ? 1 : 2;
          if (quantizedBaseW === (expStates[i] & 0x03)) alignmentMatches++;
        }

        // Safely extract small index array elements for standard number formatting string tokens
        const lossExpVal = expLoss.dataSync()[0];
        const lossBaseVal = baseLoss.dataSync()[0];
        const alignmentPercentage = ((alignmentMatches / baseShadow.length) * 100).toFixed(1);

        console.log(`[STEP ${step}/${TOTAL_STEPS}]`);
        console.log(`  |- Experimental MSE Loss : ${lossExpVal.toFixed(5)}  |  Baseline MSE Loss : ${lossBaseVal.toFixed(5)}`);
        console.log(`  |- Structural Alignment Match : ${alignmentPercentage}%`);
        console.log(`  |- State Census Matrix        : [-1: ${((countMinusOne / expStates.length) * 100).toFixed(1)}%] [0: ${((countZero / expStates.length) * 100).toFixed(1)}%] [+1: ${((countPlusOne / expStates.length) * 100).toFixed(1)}%]\n`);
      }
    });
  }

  experimentalLayer.dispose();
  baselineLayer.dispose();
}

runSymmetryStressTest();

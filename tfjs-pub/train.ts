import * as tf from "@tensorflow/tfjs";
import { FusedBitNetLayer } from "./layers/modularFusedBitNet/FusedBitNetLayer";
import { BaselineAdamBitNetLayer } from "./layers/BaselineAdamBitNetLayer";
import { TextDataPipeline } from "./TextDataPipeline";

// --- THE CALIBRATED HYPERPARAMETER BLUEPRINT ---
const BATCH_SIZE = 64;
const SEQ_LEN = 128;
const TOTAL_STEPS = 500;
const INITIAL_LR = 0.002;
const GRAD_SCALE = 15.0; // Stochastic Rounding Layer 1 scaling multiplier
const K = 0.6; // Thermal rigidity multiplier

// --- RAW CORE TARGET STRING DATA (THE TEXT CORPUS) ---
// Employs our own layer code duplicated to build a dense, highly structural syntactic matrix
const RAW_TRAINING_CORPUS = `
import * as tf from '@tensorflow/tfjs';
export class FusedBitNetLayer {
    private packedState: tf.Tensor2D;
    private gamma: number = 1.0;
    constructor(inFeatures: number, outFeatures: number) {
        const totalParams = inFeatures * outFeatures;
        const initialStates = new Int32Array(totalParams).fill(1);
        this.packedState = tf.tensor1d(initialStates, 'int32').reshape([inFeatures, outFeatures]);
    }
    public forward(x: tf.Tensor2D): tf.Tensor2D {
        const customOp = tf.customGrad((...args: any[]): any => {
            return tf.tidy(() => {
                const xInput = args as tf.Tensor2D;
                const { weight } = FusedBitNetLayer.unpackState(this.packedState);
                const realWeights = tf.sub(weight.toFloat(), tf.scalar(1.0, 'float32'));
                const output = tf.matMul(xInput, realWeights);
                return { value: tf.mul(output, tf.scalar(this.gamma)), gradFunc: (dy: any) => dy };
            });
        });
        return customOp(x);
    }
}
`.repeat(50);

async function runValidationBattle() {
  console.log("=========================================================");
  console.log("LAUNCHING DUAL-OPTIMIZER ARDUOUS LANGUAGE MODEL BATTLE  ");
  console.log("=========================================================");

  // 1. Initialize Pipeline & Vocabulary Dimensions
  const pipeline = new TextDataPipeline(RAW_TRAINING_CORPUS);
  const vocabSize = pipeline.vocabSize;
  console.log(`Dataset Length: ${RAW_TRAINING_CORPUS.length} characters`);
  console.log(`Vocabulary Size Extracted: ${vocabSize} unique characters\n`);

  // 2. Instantiate Both Contending Layers with Identical Mappings
  const experimentalLayer = new FusedBitNetLayer({
    inFeatures: vocabSize,
    outFeatures: vocabSize,
    gradScale: GRAD_SCALE,
    K,
  });
  const baselineLayer = new BaselineAdamBitNetLayer({
    inFeatures: vocabSize,
    outFeatures: vocabSize,
  });

  // --- SYSTEMS CORRECTION: RUNTIME SYMMETRY-BREAKING FOR ADAM ---
  // Overwrite the baseline's zero-initialized shadow weights with a Xavier-scaled random distribution.
  // This breaks the mathematical vacuum and lets Adam's continuous values navigate features.
  tf.tidy(() => {
    const scale = Math.sqrt(2.0 / (vocabSize + vocabSize));
    const randomWeights = tf.randomNormal([vocabSize, vocabSize], 0.0, scale, "float32");

    // Safely swap the internal baseline tensor reference to bypass editing the secondary file
    const oldWeights = (baselineLayer as any).shadowWeights;
    (baselineLayer as any).shadowWeights = tf.keep(randomWeights);
    oldWeights.dispose();
  });

  // 3. The Core Execution Training Loop
  for (let step = 1; step <= TOTAL_STEPS; step++) {
    tf.tidy(() => {
      // Calculate host-driven learning rate decay parameter (1.0 down to 0.0)
      const currentLrNormalized = Math.max(0.0, 1.0 - step / TOTAL_STEPS);
      const activeLr = INITIAL_LR * currentLrNormalized;

      // Extract an active safe vectorized sliding window batch [BATCH_SIZE, SEQ_LEN]
      const { x: xTokens, y: yTokens } = pipeline.nextBatch(BATCH_SIZE, SEQ_LEN);

      // Expand tokens into continuous one-hot vectors [BATCH_SIZE, SEQ_LEN, vocabSize]
      const xOneHot = tf.oneHot(xTokens, vocabSize).toFloat();
      const yOneHot = tf.oneHot(yTokens, vocabSize).toFloat();

      // Flatten batch/sequence dimensions into a continuous 2D matrix [8192, vocabSize]
      const xFlattened = xOneHot.reshape([BATCH_SIZE * SEQ_LEN, vocabSize]) as tf.Tensor2D;
      const yFlattened = yOneHot.reshape([BATCH_SIZE * SEQ_LEN, vocabSize]) as tf.Tensor2D;

      // --- FORWARD EXPERIMENTAL LAYER PASS & INDEPENDENT GRADIENTS ---
      const expLogits = experimentalLayer.forward(xFlattened);
      const expSoftmax = tf.softmax(expLogits);
      const expLoss = tf.neg(tf.mean(tf.sum(tf.mul(yFlattened, tf.log(tf.add(expSoftmax, 1e-12))), 1)));

      // Pure decoupled error map and parameter matrix contraction for the 1-byte system
      const expLossGradients = tf.div(tf.sub(expSoftmax, yFlattened), tf.scalar(BATCH_SIZE * SEQ_LEN, "float32"));
      const expParamGradients = tf.matMul(xFlattened.transpose(), expLossGradients) as tf.Tensor2D;

      // --- FORWARD BASELINE LAYER PASS & INDEPENDENT GRADIENTS ---
      const baseLogits = baselineLayer.forward(xFlattened);
      const baseSoftmax = tf.softmax(baseLogits);
      const baseLoss = tf.neg(tf.mean(tf.sum(tf.mul(yFlattened, tf.log(tf.add(baseSoftmax, 1e-12))), 1)));

      // Pure decoupled error map and parameter matrix contraction for the FP32 Adam baseline
      const baseLossGradients = tf.div(tf.sub(baseSoftmax, yFlattened), tf.scalar(BATCH_SIZE * SEQ_LEN, "float32"));
      const baseParamGradients = tf.matMul(xFlattened.transpose(), baseLossGradients) as tf.Tensor2D;

      // --- SYNCHRONIZED BACKWARD PATH WRITE-BACK ---
      // Both layer systems optimize independently based on their own unique forward metrics
      experimentalLayer.applyStep(expParamGradients, currentLrNormalized);
      baselineLayer.applyStep(baseParamGradients, activeLr);

      // --- 4. FORENSIC TELEMETRY DATA MONITORING DASHBOARD ---
      if (step === 1 || step % 50 === 0) {
        // Out-of-graph read block to inspect internal VRAM contents safely
        const expStates = (experimentalLayer as any).packedState.dataSync() as Int32Array;
        const baseShadow = (baselineLayer as any).shadowWeights.dataSync() as Float32Array;

        // Decode packed elements to compute distribution statistics
        let countMinusOne = 0;
        let countZero = 0;
        let countPlusOne = 0;
        let maxedPositiveInertia = 0;
        let maxedNegativeInertia = 0;

        for (let i = 0; i < expStates.length; i++) {
          const byte = expStates[i];
          // Replicate Unit 1 extraction math natively on the host loop
          const wToken = byte & 0x03;
          const unsignedMom = Math.floor(byte / 4) & 0x3f;
          const mom = unsignedMom >= 32 ? unsignedMom - 64 : unsignedMom;

          if (wToken === 0) countMinusOne++;
          if (wToken === 1) countZero++;
          if (wToken === 2) countPlusOne++;
          if (mom === 31) maxedPositiveInertia++;
          if (mom === -32) maxedNegativeInertia++;
        }

        // Evaluate baseline quantized values to match alignment delta trajectories
        let alignmentMatches = 0;
        for (let i = 0; i < baseShadow.length; i++) {
          const val = baseShadow[i];
          const quantizedBaseW = val < -0.5 ? 0 : val < 0.5 ? 1 : 2; // Map float shadow to 0, 1, 2 space
          const expW = expStates[i] & 0x03;
          if (quantizedBaseW === expW) alignmentMatches++;
        }

        const lossExpVal = expLoss.dataSync()[0];
        const lossBaseVal = baseLoss.dataSync()[0];
        const alignmentPercentage = ((alignmentMatches / baseShadow.length) * 100).toFixed(1);
        const currentThermalScale = 1.0 + K * (1.0 - currentLrNormalized);

        console.log(`[STEP ${step}/${TOTAL_STEPS}] --- Thermal Annealing Scale_t: ${currentThermalScale.toFixed(3)}`);
        console.log(`  |- Experimental Loss : ${lossExpVal.toFixed(4)}  |  Baseline Loss : ${lossBaseVal.toFixed(4)}`);
        console.log(`  |- Structural Alignment Matrix Match  : ${alignmentPercentage}% with FP32 Adam Baseline`);
        console.log(`  |- State Census Matrix                : [-1 Weights: ${((countMinusOne / expStates.length) * 100).toFixed(1)}%] [0 Weights: ${((countZero / expStates.length) * 100).toFixed(1)}%] [+1 Weights: ${((countPlusOne / expStates.length) * 100).toFixed(1)}%]`);
        console.log(`  |- Inertial Shield Saturation Pool    : [Max Positive (+31): ${maxedPositiveInertia}] [Max Negative (-32): ${maxedNegativeInertia}]\n`);
      }
    });

    // Yield execution line back to host architecture loop momentarily to guarantee UI responsiveness
    if (step % 10 === 0) {
      await tf.nextFrame();
    }
  }

  // Teardown VRAM footprints cleanly
  experimentalLayer.dispose();
  baselineLayer.dispose();
  pipeline.dispose();
  console.log("=========================================================");
  console.log("BATTLE COMPLETE. VRAM TEARDOWN ABSOLUTE. TERMINATING RUN.");
  console.log("=========================================================");
}

runValidationBattle();

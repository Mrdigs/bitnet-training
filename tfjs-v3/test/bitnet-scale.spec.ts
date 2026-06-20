import * as tf from "@tensorflow/tfjs-node";
import { expect } from "chai";
import { PersistentState } from "../lib/PersistentState";
import { ReferenceBitNetStrategy } from "../lib/strategy/ReferenceBitNetStrategy";
import { StochasticBitNetStrategy } from "../lib/strategy/StochasticBitNetStrategy";

describe.skip("BitNet Strategy Initial Forward Pass Divergence Test", () => {
  let rawWeights: tf.Tensor2D;
  let rawInputs: tf.Tensor2D;
  let refState: PersistentState;
  let stochState: PersistentState;

  // Enforce a strict shape configuration perfectly divisible by 4 for the bit-packer
  const inFeatures = 16;
  const outFeatures = 8;
  const batchSize = 4;

  beforeEach(() => {
    // Generate identical pseudo-random matrices to simulate pristine model builds
    rawWeights = tf.initializers.glorotUniform({}).apply([inFeatures, outFeatures], "float32") as tf.Tensor2D;
    rawInputs = tf.randomNormal([batchSize, inFeatures], 0, 1, "float32") as tf.Tensor2D;

    refState = new PersistentState();
    stochState = new PersistentState();
  });

  afterEach(() => {
    rawWeights.dispose();
    rawInputs.dispose();
    refState.dispose();
    stochState.dispose();
  });

  it("should match tensor properties between Reference and Stochastic strategies", () => {
    const refStrategy = new ReferenceBitNetStrategy();
    const stochStrategy = new StochasticBitNetStrategy(10.0, 0.5);

    // ----------------------------------------------------
    // STEP 1: INITIALISE WEIGHTS
    // ----------------------------------------------------
    const refPacked = refStrategy.prepareInitialWeights(rawWeights, refState);
    const stochPacked = stochStrategy.prepareInitialWeights(rawWeights, stochState);

    // ----------------------------------------------------
    // STEP 2: QUANTIZE ACTIVATIONS (Simulating Layer.call forward pass step 1)
    // ----------------------------------------------------
    const refQAct = refStrategy.quantizeActivations(rawInputs, refState);
    const stochQAct = stochStrategy.quantizeActivations(rawInputs, stochState);

    // ----------------------------------------------------
    // STEP 3: DECODE WEIGHTS (Simulating Layer.call forward pass step 2)
    // ----------------------------------------------------
    const refDecWeights = refStrategy.decodeWeights(refPacked, refState);
    const stochDecWeights = stochStrategy.decodeWeights(stochPacked, stochState);

    // ----------------------------------------------------
    // STEP 4: INTEGER MATRIX MULTIPLICATION
    // ----------------------------------------------------
    const refMatMul = tf.matMul(refQAct, refDecWeights);
    const stochMatMul = tf.matMul(stochQAct, stochDecWeights);

    // ----------------------------------------------------
    // STEP 5: DEQUANTIZE OUTPUTS
    // ----------------------------------------------------
    const refFinalOutput = refStrategy.dequantizeOutputs(refMatMul, refState);
    const stochFinalOutput = stochStrategy.dequantizeOutputs(stochMatMul, stochState);

    // ----------------------------------------------------
    // METRIC EXTRACTIONS (STEP 0)
    // ----------------------------------------------------
    const refMean = tf.mean(tf.abs(refFinalOutput)).dataSync()[0];
    const stochMean = tf.mean(tf.abs(stochFinalOutput)).dataSync()[0];

    console.log(`\n  📊 INITIALIZATION METRICS (STEP 0):`);
    console.log(`  ====================================`);
    console.log(`  Ref Beta Scale (L1):  ${refState.get("beta")?.dataSync()[0].toFixed(6)}`);
    console.log(`  Stoch Gamma (Init):   ${stochState.get("gamma")?.dataSync()[0].toFixed(6)}`);
    console.log(`  Ref Output Mean Abs:  ${refMean.toFixed(6)}`);
    console.log(`  Stoch Output Mean Abs: ${stochMean.toFixed(6)}`);

    // ASSERTION CHECK: Step 0 outputs must match completely
    const varianceRatio0 = stochMean / refMean;
    expect(varianceRatio0).to.be.closeTo(1.0, 0.01, `Step 0 variance broken! Stochastic mean magnitude is ${varianceRatio0.toFixed(2)}x different from Reference.`);

    // ----------------------------------------------------
    // STEP 6: SIMULATE AN OPTIMIZATION BACKWARD STEP
    // ----------------------------------------------------
    // Create simulated incoming gradients (dy) with matching matrix shapes
    const dummyGradients = tf.randomNormal([inFeatures, outFeatures], 0, 0.1, "float32") as tf.Tensor2D;

    // Run updates for both strategies
    const refStepUpdate = refStrategy.computeUpdate(refPacked, dummyGradients, refState, 0.01, 1);
    const stochStepUpdate = stochStrategy.computeUpdate(stochPacked, dummyGradients, stochState, 0.01);

    // ----------------------------------------------------
    // STEP 7: EVALUATE SECOND FORWARD PASS (STEP 1)
    // ----------------------------------------------------
    const refQAct2 = refStrategy.quantizeActivations(rawInputs, refState);
    const stochQAct2 = stochStrategy.quantizeActivations(rawInputs, stochState);

    const refDecWeights2 = refStrategy.decodeWeights(refStepUpdate, refState);
    const stochDecWeights2 = stochStrategy.decodeWeights(stochStepUpdate, stochState);

    const refMatMul2 = tf.matMul(refQAct2, refDecWeights2);
    const stochMatMul2 = tf.matMul(stochQAct2, stochDecWeights2);

    const refFinalOutput2 = refStrategy.dequantizeOutputs(refMatMul2, refState);
    const stochFinalOutput2 = stochStrategy.dequantizeOutputs(stochMatMul2, stochState);

    const refMean2 = tf.mean(tf.abs(refFinalOutput2)).dataSync()[0];
    const stochMean2 = tf.mean(tf.abs(stochFinalOutput2)).dataSync()[0];

    console.log(`\n  📊 POST-UPDATE METRICS (STEP 1):`);
    console.log(`  ====================================`);
    console.log(`  Ref Beta Scale (Step 1):   ${refState.get("beta")?.dataSync()[0].toFixed(6)}`);
    console.log(`  Stoch Gamma (Step 1):      ${stochState.get("gamma")?.dataSync()[0].toFixed(6)}`);
    console.log(`  Ref Output Mean Abs:       ${refMean2.toFixed(6)}`);
    console.log(`  Stoch Output Mean Abs:     ${stochMean2.toFixed(6)}`);
    console.log(`  Scale Variance Shift Ratio: ${(stochMean2 / refMean2).toFixed(4)}x\n`);

    // Clean up temporary execution tape buffers
    refPacked.dispose();
    stochPacked.dispose();
    refQAct.dispose();
    stochQAct.dispose();
    refDecWeights.dispose();
    stochDecWeights.dispose();
    refMatMul.dispose();
    stochMatMul.dispose();
    refFinalOutput.dispose();
    stochFinalOutput.dispose();
    dummyGradients.dispose();
    refStepUpdate.dispose();
    stochStepUpdate.dispose();
    refQAct2.dispose();
    stochQAct2.dispose();
    refDecWeights2.dispose();
    stochDecWeights2.dispose();
    refMatMul2.dispose();
    stochMatMul2.dispose();
    refFinalOutput2.dispose();
    stochFinalOutput2.dispose();
  });
});

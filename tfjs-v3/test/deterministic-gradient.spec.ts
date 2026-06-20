import * as tf from "@tensorflow/tfjs-node";
import { expect } from "chai";
import { PersistentState } from "../lib/PersistentState";
import { StochasticBitNetStrategy } from "../lib/strategy/StochasticBitNetStrategy";
import { MiniBitNetStrategy } from "../lib/strategy/MiniBitNetStrategy";

describe.skip("Deterministic Single-Cell 15-Step Optimization Tracking", () => {
  let state: PersistentState;
  const fixedSeed = 42; // Enforces absolute mathematical reproducibility

  // Set the precise boundary dimensions required for a single-column weight trace
  const inFeatures = 4;
  const outFeatures = 1;
  const targetRowIndex = 1; // Explicit coordinate cell we will interrogate behind the scenes

  beforeEach(() => {
    state = new PersistentState();
  });

  afterEach(() => {
    state.dispose();
  });

  it("should trace the exact register lifecycle of a single weight cell across 15 violent update steps", () => {
    // Calibrated with default strategy parameters
    // const strategy = new StochasticBitNetStrategy(1, 1, fixedSeed);
    const strategy = new MiniBitNetStrategy(fixedSeed);

    // Initialize a 4x1 matrix to a clean neutral state (Storage ID 1, mapping to 0.0)
    let packedWeightTensor = strategy.pack(tf.fill([inFeatures, outFeatures], 1, "int32"), tf.zeros([inFeatures, outFeatures], "int32"));

    // Build an aggressive 15-step multi-phase gradient stream matching your exact conditions
    const gradientStream = [
      // PHASE 1: Sustained Positive Push (Weight MUST move down toward -1.0)
      tf.fill([inFeatures, outFeatures], 0.5, "float32"),
      tf.fill([inFeatures, outFeatures], 0.5, "float32"),
      tf.fill([inFeatures, outFeatures], 0.5, "float32"),
      tf.fill([inFeatures, outFeatures], 0.5, "float32"),

      // PHASE 2: Violent Whip-Around Shock (Hard brake, then weight climbing UP toward +1.0)
      tf.fill([inFeatures, outFeatures], -0.8, "float32"),
      tf.fill([inFeatures, outFeatures], -0.8, "float32"),
      tf.fill([inFeatures, outFeatures], -0.8, "float32"),
      tf.fill([inFeatures, outFeatures], -0.8, "float32"),

      // PHASE 3: Zero-Gradient Dormant Hold (Velocity should decay while weight stands still)
      tf.fill([inFeatures, outFeatures], 0.0, "float32"),
      tf.fill([inFeatures, outFeatures], 0.0, "float32"),
      tf.fill([inFeatures, outFeatures], 0.0, "float32"),

      // PHASE 4: Low-Energy Thermodynamic Cooling Noise (Observe stationary register stabilization)
      tf.fill([inFeatures, outFeatures], 0.04, "float32"),
      tf.fill([inFeatures, outFeatures], -0.04, "float32"),
      tf.fill([inFeatures, outFeatures], 0.04, "float32"),
      tf.fill([inFeatures, outFeatures], 0.0, "float32"),
    ];

    console.log(`\n  ⚡ DETERMINISTIC INDIVIDUAL CELL TRACE LOGS (Row Index: ${targetRowIndex}):`);
    console.log(`  ==============================================================`);

    // Log the true Step 0 initialization starting baseline
    const initGamma = state.get("beta")?.dataSync()[0] ?? 0.0;
    const { weight: w0, momentum: m0 } = (strategy as any).unpack(packedWeightTensor);
    // const decW0 = tf.sub(w0.toFloat(), tf.scalar(1.0, "float32")).dataSync();
    // const mom0Data = strategy.decodeMomentum(m0).dataSync();
    const decW0 = w0.dataSync();
    const mom0Data = m0.dataSync();

    console.log(`  Step 0  | Init     | Grad Value:  0.00 | ` + `Weight Cell: ${decW0[targetRowIndex] >= 0 ? " " : ""}${decW0[targetRowIndex].toFixed(2)} | ` + `Mom Cell: ${mom0Data[targetRowIndex] >= 0 ? " " : ""}${mom0Data[targetRowIndex].toString().padEnd(3)} | ` + `Gamma: ${initGamma?.toFixed(4)}`);
    w0.dispose();
    m0.dispose();

    // Run the complete multi-step optimization pipeline loop
    for (let i = 0; i < gradientStream.length; i++) {
      const stepGradient = gradientStream[i];
      const learningRate = 1;

      // Extract the exact gradient value applied to our specific cell target index
      const rawGradArray = stepGradient.dataSync();
      const currentCellGrad = rawGradArray[targetRowIndex];

      // Execute a single step update mutation chain
      const nextPackedState = strategy.computeUpdate(packedWeightTensor, stepGradient, state, learningRate);

      // Unpack register layers for high-fidelity debugging of our target cell
      const { weight, momentum } = (strategy as any).unpack(nextPackedState);
      // const decodedW = tf.sub(weight.toFloat(), tf.scalar(1.0, "float32")).dataSync();
      // const momentumData = strategy.decodeMomentum(momentum).dataSync();
      const decodedW = weight.dataSync();
      const momentumData = momentum.dataSync();
      const currentGamma = state.get("beta")?.dataSync()[0] ?? 0.0;

      const stepNum = (i + 1).toString().padEnd(2);

      let phaseLabel = "Cooling ";
      if (i < 4) phaseLabel = "Pos Push";
      else if (i < 8) phaseLabel = "V-Shock ";
      else if (i < 11) phaseLabel = "Z-Hold  ";

      console.log(`  Step ${stepNum} | ${phaseLabel} | ` + `Grad Value: ${currentCellGrad >= 0 ? " " : ""}${currentCellGrad.toFixed(2)} | ` + `Weight Cell: ${decodedW[targetRowIndex] >= 0 ? " " : ""}${decodedW[targetRowIndex].toFixed(2)} | ` + `Mom Cell: ${momentumData[targetRowIndex] >= 0 ? " " : ""}${momentumData[targetRowIndex].toString().padEnd(3)} | ` + `Gamma: ${currentGamma?.toFixed(4)}`);

      // Cycle tensors back into the loop securely
      packedWeightTensor.dispose();
      packedWeightTensor = nextPackedState;

      weight.dispose();
      momentum.dispose();
      stepGradient.dispose();
    }

    // Final memory cleanup
    packedWeightTensor.dispose();
  });
});

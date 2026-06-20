import * as tf from "@tensorflow/tfjs-node";
import { FourTo1BitPackingStrategy } from "./FourTo1BitPackingStrategy";
import { PersistentState } from "../PersistentState";

export class MiniBitNetStrategy extends FourTo1BitPackingStrategy {
  private readonly seed: number | undefined;

  constructor(seed?: number) {
    super();
    this.seed = seed;
  }

  public prepareInitialWeights(rawFloatWeights: tf.Tensor, state: PersistentState): tf.Tensor {
    return tf.tidy(() => {
      // this previous version introduced bias by scaling -2 to 2
      // const beta = tf.maximum(tf.mean(tf.abs(rawFloatWeights)), tf.scalar(1e-5));
      const beta = tf.maximum(tf.max(tf.abs(rawFloatWeights)), tf.scalar(1e-5));

      const scaled = tf.div(rawFloatWeights, beta);
      const ternary = tf.clipByValue(tf.round(scaled), -1, 1);
      const residuals = tf.sub(scaled, ternary);
      return this.pack(ternary, residuals);
    });
  }

  public decodeWeights(packedTensor: tf.Tensor, state: PersistentState): tf.Tensor {
    return tf.tidy(() => {
      const { weight, residual } = this.unpack(packedTensor);
      const continuous = tf.add(weight, residual);
      // const maxBound = 1.0 + TARGET_SCALE;
      // const normalizedLatent = tf.div(latentWeights, tf.scalar(maxBound));
      const normalizedLatent = tf.div(continuous, tf.scalar(4.0));
      const beta = tf.maximum(tf.mean(tf.abs(normalizedLatent)), tf.scalar(1e-5));
      // const beta = tf.maximum(tf.mean(tf.abs(continuous)), tf.scalar(1e-5));
      state.set("beta", beta);
      return weight;
    });
  }

  public computeUpdate(weightTensor: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number, currentStep: number): tf.Tensor {
    return tf.tidy(() => {
      const { weight, residual } = this.unpack(weightTensor);
      const updatedResidual = tf.sub(residual, tf.mul(gradient, tf.tensor(learningRate)));
      const updated = this.applyStochasticRounding(weight, updatedResidual);
      return this.pack(updated.ternary, updated.residual);
    });
  }

  public quantizeActivations(inputs: tf.Tensor, state: PersistentState): tf.Tensor {
    const axis = inputs.rank - 1;

    // RMSNorm Phase
    const meanSquare = tf.mean(tf.square(inputs), axis, true);
    const rms = tf.sqrt(tf.add(meanSquare, tf.scalar(1e-5)));
    const xNorm = tf.div(inputs, rms);

    // Dynamic per-token scaling factor (Eta)
    const eta = tf.maximum(tf.max(tf.abs(xNorm), axis, true), tf.scalar(1e-5));
    state.set("eta", eta);

    return tf.customGrad((...args: any[]) => {
      const xIn = args[0] as tf.Tensor;
      const scaled = tf.mul(xIn, tf.div(tf.scalar(127.0, "float32"), eta));
      const quant = tf.clipByValue(tf.round(scaled), -127.0, 127.0);
      return {
        value: quant,
        gradFunc: (dy: tf.Tensor) => [dy],
      };
    })(xNorm);
  }

  public dequantizeOutputs(rawOutputs: tf.Tensor, state: PersistentState): tf.Tensor {
    const eta = state.get("eta");
    const beta = state.get("beta");

    if (!eta || !beta) {
      throw new Error("Dequantization failed: Context scales are missing.");
    }

    // 1. Extract the raw numerical value arrays directly out of the tracking instances
    const etaVal = eta.dataSync();
    const betaVal = beta.dataSync();

    // console.log(betaVal);

    // 2. Instantiate fresh, disconnected constant tensors from the primitive arrays.
    // Because they are new allocations, the gradient tape will completely ignore them.
    const staticEta = tf.tensor(etaVal, eta.shape, eta.dtype);
    const staticBeta = tf.tensor(betaVal, beta.shape, beta.dtype);

    // 3. Compute output scaling using the detached constants
    const rescaled = tf.mul(rawOutputs, tf.div(tf.mul(staticEta, staticBeta), tf.scalar(127.0, "float32")));

    // 4. Dispose of the temporary detached tensors immediately to prevent VRAM accumulation
    staticEta.dispose();
    staticBeta.dispose();

    return rescaled;
  }

  public applyUpdate(weightVar: tf.Variable, update: tf.Tensor): void {
    weightVar.assign(update);
  }

  public applyStochasticRounding(ternary: tf.Tensor, residual: tf.Tensor): { ternary: tf.Tensor; residual: tf.Tensor } {
    //return { ternary, residual };
    return tf.tidy(() => {
      // 1. Vectorized Truncation (round towards zero to extract pure integer shift)
      const integerShift = tf.where(tf.greaterEqual(residual, 0), tf.floor(residual), tf.ceil(residual));

      // 2. Extract fractional remainder
      const fractionalResidual = tf.sub(residual, integerShift);

      // 3. Vectorized Stochastic logic using the fixed instance seed
      // ALERT!!!! THIS IS NO LONGER STACHASTIC WITH 0.5/0.5 - JUST TESTING, DO NO COMMIT!
      // BUT IT WORKS BETTER!
      //const randomThresholds = tf.randomUniform(residual.shape, 0, 1, "float32", this.seed);
      //const randomThresholds = tf.randomUniform(residual.shape, 0.333, 1, "float32", this.seed);
      // Wait what? if a < b, right.... - RIGHT NOW, IS SHOULDNT EVER - RIGHT?
      const randomThresholds = tf.randomUniform(residual.shape, 0.5, 0.5, "float32", this.seed);
      //const randomThresholds = this.getFastScrambledThresholds(residual.shape, this.step++);
      const stochasticFlip = tf.less(randomThresholds, tf.abs(fractionalResidual));
      const stochasticDirection = tf.sign(fractionalResidual);
      const stochasticShift = tf.where(stochasticFlip, stochasticDirection, tf.zerosLike(residual));

      // 4. Calculate proposed state changes and enforce boundary clamping [-1, 1]
      const totalProposedShift = tf.add(integerShift, stochasticShift);
      const proposedTernary = tf.add(ternary, totalProposedShift);
      const updatedTernary = tf.clipByValue(proposedTernary, -1, 1);

      // 5. Calculate exactly how much shift actually executed after boundaries
      const totalActualShift = tf.sub(updatedTernary, ternary);

      // 6. Anti Flip-Flop Masking Loopless Logic:
      // Track where the discrete state actually successfully changed values
      const ternaryChanged = tf.notEqual(updatedTernary, ternary);

      // Calculate base remainder by subtracting the allowed shift from original residual
      const residualMinusActualShift = tf.sub(residual, totalActualShift);

      // Vectorized Truncation (round toward zero) for the updated residual path
      const truncatedResidualMinusActual = tf.where(tf.greaterEqual(residualMinusActualShift, 0), tf.floor(residualMinusActualShift), tf.ceil(residualMinusActualShift));

      // If ternary changed, truncate fractions away. If unchanged, preserve full fraction.
      const updatedResidual = tf.where(
        ternaryChanged,
        truncatedResidualMinusActual, // Keeps remaining unapplied integer overrides
        residualMinusActualShift, // Preserves original fractional magnitude to try again
      );

      // original where it crossed the boundary of 0. might need this if I want
      // to explore the diminishing toward zero strategy at some point
      // const updatedResidual = tf.sub(residual, actualShiftAllowed);

      return {
        ternary: updatedTernary,
        residual: updatedResidual,
      };
    });
  }

  public override unpack(packed: tf.Tensor): { weight: tf.Tensor; residual: tf.Tensor } {
    return tf.tidy(() => {
      const { weight, residual } = super.unpack(packed);
      const scaled = this.scale(residual, this.SCALING.targetScale, this.SCALING.exponent);
      const ternary = tf.sub(weight.toFloat(), tf.scalar(1.0, "float32"));
      return { weight: ternary, residual: scaled };
    });
  }

  public override pack(weight: tf.Tensor, residual: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const raw = this.unscale(residual, this.SCALING.targetScale, this.SCALING.exponent);
      //console.log("Packing,", residual.dataSync().slice(0, 20));
      //console.log("Unscaled,", raw.dataSync().slice(0, 20));
      //console.log("Rescaled,", this.scale(raw, 3, 5).dataSync().slice(0, 20));
      //process.exit(1);
      const floor = tf.floor(raw);
      const remainder = tf.sub(raw, floor);
      //const randomThresholds = tf.randomUniform(raw.shape, 0, 1, "float32", this.seed);
      const randomThresholds = this.getFastScrambledThresholds(raw.shape, this.step++);
      //console.log("\n", randomThresholds.dataSync().slice(0, 20), randomThresholds.shape, remainder.shape);
      const roundUpBit = tf.cast(tf.less(randomThresholds, remainder), "float32");
      const unscaled = tf.add(floor, roundUpBit);
      const ternary = tf.add(weight.toInt(), tf.scalar(1));
      return super.pack(ternary, unscaled);
    });
  }

  private step = 0;

  // private SCALING = { targetScale: 3, exponent: 3 }; // Loss 2,21 / Acc 0.1822
  // private SCALING = { targetScale: 5, exponent: 6 }; // Loss 2.10 / Acc 0.3241
  // private SCALING = { targetScale: 4, exponent: 3 }; // Loss 2.09 / Acc 0.2758
  private SCALING = { targetScale: 3, exponent: 5 }; // Loss 1.99 / Acc 0.32
  // private SCALING = { targetScale: 4, exponent: 5 }; // Loss 1.89 / Acc 0.35

  /**
   * Reverses the non-linear polynomial compression back to the raw scale relative to a capacity of 31.
   *
   * @param scaled The compressed tensor to expand.
   * @param targetScale The original output multiplier used to scale (e.g., 6).
   * @param exponent The original polynomial power used to scale (e.g., 3).
   */
  public unscale(scaled: tf.Tensor, targetScale: number, exponent: number): tf.Tensor {
    return tf.tidy(() => {
      const sign = tf.sign(scaled);
      const normalizedY = tf.div(tf.abs(scaled), tf.scalar(targetScale));
      const rootValue = tf.pow(normalizedY, tf.scalar(1 / exponent));
      return tf.mul(tf.scalar(31), tf.mul(sign, rootValue));
    });
  }

  /**
   * Compresses values using a non-linear polynomial curve relative to a capacity of 31.
   *
   * @param unscaled The input tensor to compress.
   * @param targetScale The output multiplier/amplitude at peak capacity (e.g., 6).
   * @param exponent The polynomial power that controls sharpening/sparsity (e.g., 3 for cubic).
   */
  public scale(unscaled: tf.Tensor, targetScale: number, exponent: number): tf.Tensor {
    return tf.tidy(() => {
      const normalized = tf.div(unscaled, tf.scalar(31));
      return tf.mul(tf.scalar(targetScale), tf.pow(normalized, exponent));
    });
  }

  /**
   * Fast MurmurHash3 fmix32 bit-scrambler shortcut.
   * Generates an instantly distributed, time-varying threshold matrix on the hardware level.
   */
  private getFastScrambledThresholds(shape: number[], step: number): tf.Tensor {
    const totalElements = shape.reduce((a, b) => a * b, 1);
    const noiseBuffer = new Float32Array(totalElements);

    // Run the hardware bitwise loop natively on the processor
    for (let i = 0; i < totalElements; i++) {
      // 1. Interleave the base coordinate with the time step
      let h = (i + 1) ^ step;

      // 2. The MurmurHash3 fmix32 bit-scrambler constants
      h ^= h >>> 16;
      h = Math.imul(h, 0x85ebca6b); // Safe 32-bit integer multiplication in JS
      h ^= h >>> 13;
      h = Math.imul(h, 0xc2b2ae35);
      h ^= h >>> 16;

      // 3. Map the scrambled bits to a clean 0.0 -> 1.0 float range
      // 0x7FFFFFFF isolates the positive range of a 32-bit signed int
      noiseBuffer[i] = (h & 0x7fffffff) / 2147483647.0;
    }

    // Pack the typed array directly into a WebGL tensor and shape it
    return tf.tensor(noiseBuffer, shape, "float32");
  }
}

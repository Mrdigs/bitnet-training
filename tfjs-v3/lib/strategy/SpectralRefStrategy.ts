import * as tf from "@tensorflow/tfjs-node";
import { ReferenceBitNetStrategy } from "./ReferenceBitNetStrategy";
import { PersistentState } from "../PersistentState";

export class SpectralRefStrategy extends ReferenceBitNetStrategy {
  /**
   * Compatible Sandbox Implementation of the Hadamard 1-Bit Optimizer
   */
  public computeUpdate(weight: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number, currentStep: number): tf.Tensor {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    return tf.tidy(() => {
      const originalShape = weight.shape;
      const is2D = originalShape.length >= 2;
      const originalRows = is2D ? originalShape[0]! : 1;
      const originalCols = is2D ? originalShape[1]! : originalShape[0]!;

      // --- STEP 1: CALCULATE NEXT POWER OF TWO FOR PADDING ---
      // For MNIST (10 rows), nextPowerOfTwo becomes 16
      const nextPowerOfTwo = Math.pow(2, Math.ceil(Math.log2(originalRows)));
      const padAmount = nextPowerOfTwo - originalRows;

      let paddedGradient = gradient;
      if (padAmount > 0 && is2D) {
        // Pad the rows with zeros at the bottom of the matrix
        paddedGradient = tf.pad2d(
          gradient as tf.Tensor2D,
          [
            [0, padAmount],
            [0, 0],
          ],
          0,
        );
      }

      // --- STEP 2: ROW-WISE VARIANCE SENSING (Using original dimensions) ---
      const rowVarianceShape = is2D ? [originalRows, 1] : [originalRows];
      const secondMomentRow = state.getOrCreate("v_row", () => tf.zeros(rowVarianceShape));

      const gradSquared = tf.square(gradient);
      const rowGradMean = is2D ? tf.mean(gradSquared, 1, true) : tf.mean(gradSquared);
      const nextVRow = tf.add(tf.mul(secondMomentRow, beta2), tf.mul(rowGradMean, 1 - beta2));
      const vHatRow = tf.div(nextVRow, tf.scalar(1 - Math.pow(beta2, currentStep + 1)));

      // --- STEP 3: SPATIAL ROTATION VIA HADAMARD TRANSFORM ---
      // Operates safely on the padded 16-row matrix
      const rotatedGradients = fastWalshHadamardTransform(paddedGradient);

      // --- STEP 4: FIRST MOMENT UPDATE IN FREQUENCY DOMAIN ---
      // The tracking buffer must match the padded shape [16, Columns]
      const firstMomentHadamard = state.getOrCreate("m_hadamard", () => tf.zerosLike(paddedGradient));
      const nextMHadamard = tf.add(tf.mul(firstMomentHadamard, beta1), tf.mul(rotatedGradients, 1 - beta1));
      const mHatHadamard = tf.div(nextMHadamard, tf.scalar(1 - Math.pow(beta1, currentStep + 1)));

      // Extract 1-bit frequency direction signs
      const directionSigns = tf.sign(mHatHadamard);

      // --- STEP 5: ROTATION INVERSION ---
      const inverseRotatedDirection = fastWalshHadamardTransform(directionSigns);

      // --- STEP 6: SLICE BACK DOWN TO MNIST DIMENSIONS ---
      let finalDirection = inverseRotatedDirection;
      if (padAmount > 0 && is2D) {
        // Slice out just the active 10 rows, dropping the 6 padding rows
        finalDirection = tf.slice2d(inverseRotatedDirection as tf.Tensor2D, [0, 0], [originalRows, originalCols]);
      }

      // --- STEP 7: EXECUTE STEP WEIGHT UPDATE ---
      const updateDelta = tf.div(finalDirection, tf.add(tf.sqrt(vHatRow), tf.scalar(eps)));
      const nextWeight = tf.sub(weight, tf.mul(updateDelta, tf.scalar(learningRate)));

      // --- STEP 8: KEEP VARIABLES PERSISTENT ---
      state.set("m_hadamard", tf.keep(nextMHadamard));
      state.set("v_row", tf.keep(nextVRow));

      firstMomentHadamard.dispose();
      secondMomentRow.dispose();

      return nextWeight;
    });
  }
}

/**
 * Computes an In-Place Fast Walsh-Hadamard Transform (FWHT) on a 1D or 2D Tensor.
 * Uses an O(N log N) iterative butterfly network structure.
 */
function fastWalshHadamardTransform(tensor: tf.Tensor): tf.Tensor {
  return tf.tidy(() => {
    const shape = tensor.shape;
    const is2D = shape.length >= 2;
    const rows = is2D ? shape[0]! : 1;
    const cols = is2D ? shape[1]! : shape[0]!;

    // Enforce power-of-two constraints for radix-2 butterfly pairing
    if ((cols & (cols - 1)) !== 0) {
      throw new Error(`Hadamard transformation requires column dimension to be a power of 2. Found: ${cols}`);
    }

    // Force shape layout into a predictable 2D matrix [Rows, Cols]
    let X = tf.reshape(tensor, [rows, cols]);
    const steps = Math.log2(cols);

    // Pure GPU Butterfly Execution Network
    for (let step = 0; step < steps; step++) {
      const groupSize = Math.pow(2, step);
      const halfGroup = groupSize;

      // Reshape the columns to dynamically isolate pairs for addition and subtraction
      // This groups elements so we can compute sums and differences simultaneously via broadcasting
      const numGroups = cols / (groupSize * 2);
      X = tf.reshape(X, [rows * numGroups, 2, groupSize]);

      // Split into Top (Left) and Bottom (Right) elements of the butterfly node
      const topValues = tf.slice(X, [0, 0, 0], [-1, 1, -1]).reshape([rows * numGroups, groupSize]);
      const bottomValues = tf.slice(X, [0, 1, 0], [-1, 1, -1]).reshape([rows * numGroups, groupSize]);

      // Apply the core Walsh-Hadamard Kernel matrix transformation:
      // [1  1] * [top   ]  =  [top + bottom]
      // [1 -1]   [bottom]     [top - bottom]
      const nextTop = tf.add(topValues, bottomValues);
      const nextBottom = tf.sub(topValues, bottomValues);

      // Stack them back together sequentially along the grouping axis
      X = tf.stack([nextTop, nextBottom], 1);
    }

    // Flatten back to standard matrix space and scale by 1 / sqrt(N) to conserve energy
    let out = tf.reshape(X, [rows, cols]).div(Math.sqrt(cols));
    return tf.reshape(out, shape);
  });
}

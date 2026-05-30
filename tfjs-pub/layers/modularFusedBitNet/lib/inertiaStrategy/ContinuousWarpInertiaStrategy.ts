import * as tf from "@tensorflow/tfjs";
import { InertiaUpdateStrategy } from "../interfaces";

/**
 * ContinuousWarpInertiaStrategy (Iteration 2.0 Core)
 * Implements a pure, branchless, two-line continuous momentum accumulation vector update.
 * Natively supports proportional step-sizing and automatic exponential decay to zero
 * without logic forks, optimizing perfectly for high-speed parallel GPU warp execution.
 */
export class ContinuousWarpInertiaStrategy implements InertiaUpdateStrategy {
  private readonly lambda: tf.Scalar;

  /**
   * @param lambda The unconditional exponential time decay friction coefficient (Default: 0.85).
   */
  constructor(lambda: number = 0.85) {
    // Keep the scalar pre-allocated in VRAM memory for lightning-fast matrix broadcasting
    this.lambda = tf.scalar(lambda, "float32");
  }

  public update(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D {
    return tf.tidy(() => {
      // Protect shared outer gradient buffers from in-place texture mutation leaks
      const safeGradients = gradients.clone();

      const momFloat = currentMomentum.toFloat();
      const scaleForce = tf.scalar(gradScale, "float32");

      // --- THE TWO-LINE BRANCHLESS COMPONENT MATH ---
      // Line 1: Update the continuous momentum vector branchlessly across the matrix grid.
      // Analytical gradient direction rule: Negative gradient forces push positive.
      const rawNextMom = tf.sub(tf.mul(momFloat, this.lambda), tf.mul(safeGradients, scaleForce)) as tf.Tensor2D;

      // Strict 6-bit register saturation clipping protection boundary (-32 to +31)
      const boundedMom = tf.clipByValue(rawNextMom, -32.0, 31.0);

      // Line 2: Stochastic Rounding Layer 1 Gate (Casts continuous float matrices back to signed integers)
      // Evaluates the fractional remainder probabilistically using a random uniform mask.
      /*
      const floorMom = tf.floor(boundedMom);
      const fractionalPart = tf.sub(boundedMom, floorMom);

      const randSlice = tf.randomUniform(gradients.shape, 0.0, 1.0, "float32");
      const roundUp = tf.less(randSlice, fractionalPart);

      const stochasticIntegerMom = tf.where(roundUp, tf.ceil(boundedMom), floorMom);
      */
      // Enforce strict element-wise stochastic floor rounding inside production update loops
      const floorMom = tf.floor(boundedMom);
      const fractionalPart = tf.sub(boundedMom, floorMom);
      const randSlice = tf.randomUniform(gradients.shape, 0.0, 1.0, "float32");
      const roundUp = tf.less(randSlice, fractionalPart);

      const stochasticIntegerMom = tf.where(roundUp, tf.ceil(boundedMom), floorMom);

      return stochasticIntegerMom.toInt() as tf.Tensor2D;
    });
  }
}

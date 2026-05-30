import * as tf from "@tensorflow/tfjs";
import { ForwardProjectionQuantizer } from "../interfaces";

/**
 * SymmetricalVarianceQuantizer (Iteration 2.1 Core)
 * Resolved: Fixed dynamic array tracking signatures by explicitly targeting
 * index [0] on .dataSync() buffers to satisfy strict typescript numeric contracts.
 */
export class SymmetricalVarianceQuantizer implements ForwardProjectionQuantizer {
  private readonly initialSigma: number;

  /**
   * @param initialSourceWeights The master continuous random weights matrix used to synchronize scale at birth.
   */
  constructor(initialSourceWeights: tf.Tensor2D) {
    // Enforce strict out-of-graph read to lock down our baseline initialization scale multiplier
    const meanVal = tf.mean(initialSourceWeights);
    const stdDev = tf.sqrt(tf.mean(tf.square(tf.sub(initialSourceWeights, meanVal))));

    // FIX A: Target index [0] to extract the raw primitive number from the Float32Array buffer
    const sigmaData = stdDev.dataSync()[0];

    this.initialSigma = sigmaData > 0 ? sigmaData : 1.0;
  }

  public transformWeights(weightTokens: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      // Strict 2-bit ternary token enforcement safety rail (maps 3 down to 2)
      const safelyClamped = tf.clipByValue(weightTokens, 0, 2);
      // Map internal unsigned tokens (0, 1, 2) straight to mathematical floats (-1.0, 0.0, +1.0)
      return tf.sub(safelyClamped.toFloat(), tf.scalar(1.0, "float32"));
    });
  }

  public calculateScale(weightTokens: tf.Tensor2D): number {
    return tf.tidy(() => {
      const realWeights = this.transformWeights(weightTokens);
      const averageMagnitude = tf.mean(tf.abs(realWeights));

      // FIX B: Target index [0] here as well to cleanly collapse the data buffer row
      const gammaData = averageMagnitude.dataSync()[0];

      // Multiply our standard running average magnitude by our initial scaling sigma ratio
      return gammaData > 0 ? gammaData * this.initialSigma : this.initialSigma;
    });
  }
}

import * as tf from "@tensorflow/tfjs-node";

export interface IBitNetStrategy {
  /**
   * Translates standard full-precision continuous weights into the exact
   * bit-packed configuration required by the strategy (setting starting velocities to 0).
   */
  prepareInitialWeights(rawFloatWeights: tf.Tensor): tf.Tensor;

  /**
   * Decomposes the packed 8-bit integer matrix into its logical components.
   */
  unpack(packedTensor: tf.Tensor): {
    ternaryWeights: tf.Tensor;
    velocity: tf.Tensor;
  };

  /**
   * Processes the entire mutation lifecycle in a single vectorized pass.
   */
  mutate(
    currentVelocity: tf.Tensor,
    currentTernaryWeights: tf.Tensor,
    gradient: tf.Tensor,
    learningRate: number,
  ): {
    finalVelocity: tf.Tensor;
    finalTernaryWeights: tf.Tensor;
  };

  /**
   * Serializes the discrete components back into a unified 8-bit state.
   */
  pack(velocity: tf.Tensor, ternaryWeights: tf.Tensor): tf.Tensor;
}

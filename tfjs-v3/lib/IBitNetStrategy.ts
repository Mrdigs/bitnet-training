import * as tf from "@tensorflow/tfjs-node";

export interface IBitNetStrategy {
  /**
   * Calculates the exact internal tracking layout shape required
   * to house the compressed variables.
   */
  getPackedShape(outFeatures: number, inFeatures: number): [number, number];

  /**
   * Transforms an initial full-precision [Out, In] floating point matrix
   * into the strategy's internal packed layout container format.
   */
  prepareInitialWeights(rawFloatWeights: tf.Tensor2D): tf.Tensor;

  /**
   * Decodes the unified float32 container storage into an uncompressed,
   * high-precision float32 matrix containing the exact ternary values (-1.0, 0.0, 1.0)
   * ready for framework mathematical execution.
   */
  getTernaryWeights(packedTensor: tf.Tensor): tf.Tensor;

  /**
   * Accepts the unified float32 layer container and the uncompressed gradients,
   * handles its own unpacking/repacking mechanics, and returns an updated unified float32 tensor.
   */
  applyGradientUpdate(currentPackedContainer: tf.Tensor, gradient: tf.Tensor, learningRate: number): tf.Tensor;
}

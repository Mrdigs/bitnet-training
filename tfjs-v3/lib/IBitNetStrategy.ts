import * as tf from "@tensorflow/tfjs";
import { PersistentState } from "./PersistentState";

export interface IBitNetStrategy {
  /**
   * Defines the target layout dimensions for the internal kernel storage matrix.
   */
  getPackedShape(outFeatures: number, inFeatures: number): tf.Shape;

  /**
   * Transforms raw float baseline initial weights into the strategy's custom configuration layout.
   */
  prepareInitialWeights(rawFloatWeights: tf.Tensor): tf.Tensor;

  /**
   * Straight-Through Estimator forward decoder pass.
   * Maps packed tracking parameters back to real floating-point tensor dimensions.
   */
  decodeWeights(packedTensor: tf.Tensor, state: PersistentState): tf.Tensor;

  /**
   * Quantization phase for input activations.
   */
  quantizeActivations(inputs: tf.Tensor, state: PersistentState): tf.Tensor;

  /**
   * Universal dequantization hook executed after matrix multiplication to restore feature variance.
   */
  dequantizeOutputs(rawOutputs: tf.Tensor, state: PersistentState): tf.Tensor;

  /**
   * Custom out-of-bounds weight state optimization step.
   */
  computeUpdate(weightVar: tf.Variable, gradient: tf.Tensor, state: PersistentState): void;
}

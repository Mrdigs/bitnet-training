import * as tf from "@tensorflow/tfjs";
import { OptimizerState } from "./OptimizerState";

export interface IBitNetStrategy {
  /**
   * Informs the model framework what shape it wants the underlying
   * weight matrix variable container to be inside the layer structure.
   */
  getPackedShape(outFeatures: number, inFeatures: number): tf.Shape;

  /**
   * Decodes your internal storage layout block into high-precision ternary matrices.
   */
  decodeWeights(packedTensor: tf.Tensor): tf.Tensor;

  /**
   * CLEAN STRATEGY MATH UPDATE
   * Executes your custom optimization mechanics. Mutates weights via `weightVar.assign()`
   * and alters tracking variables via the provided state container instance.
   */
  computeUpdate(weightVar: tf.Variable, gradient: tf.Tensor, state: OptimizerState): void;
}

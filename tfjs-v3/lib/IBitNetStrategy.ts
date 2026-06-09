import * as tf from "@tensorflow/tfjs";
import { PersistentState } from "./PersistentState";

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
   * FORWARD PASS: Receives the layer's dedicated persistent state container.
   * The strategy can now declare and mutate its own custom normalization scale variables in place.
   */
  quantizeActivations(inputs: tf.Tensor, layerState: PersistentState): tf.Tensor;

  /**
   * Executes your custom optimization mechanics. Mutates weights via `weightVar.assign()`
   * and alters tracking variables via the provided state container instance.
   */
  computeUpdate(weightVar: tf.Variable, gradient: tf.Tensor, optimizerState: PersistentState): void;
}

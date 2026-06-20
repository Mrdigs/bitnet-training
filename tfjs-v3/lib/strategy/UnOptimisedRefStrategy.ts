import * as tf from "@tensorflow/tfjs-node";
import { ReferenceBitNetStrategy } from "./ReferenceBitNetStrategy";
import { PersistentState } from "../PersistentState";

export class UnOptimisedRefStrategy extends ReferenceBitNetStrategy {
  public override computeUpdate(weight: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number, currentStep: number): tf.Tensor {
    const nextWeight = tf.sub(weight, tf.mul(gradient, tf.scalar(learningRate)));
    return nextWeight;
  }
}

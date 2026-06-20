import * as tf from "@tensorflow/tfjs-node";
import { PersistentState } from "../PersistentState";
import { MiniBitNetStrategy } from "./MiniBitNetStrategy";

export class AdamMiniBitNetStrategy extends MiniBitNetStrategy {
  public computeUpdate(weightTensor: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number, currentStep: number): tf.Tensor {
    return tf.tidy(() => {
      const { weight, momentum: residual } = this.unpack(weightTensor);
      const optimizedGradient = this.optimizeGradient(weight, gradient, state, currentStep);
      const updatedResidual = tf.sub(residual, tf.mul(optimizedGradient, tf.scalar(learningRate)));
      const updated = this.applyStochasticRounding(weight, updatedResidual);
      return this.pack(updated.ternary, updated.residual);
    });
  }

  public optimizeGradient(weight: tf.Tensor, gradient: tf.Tensor, state: PersistentState, currentStep: number): tf.Tensor {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    const firstMoment = state.getOrCreate("m", () => tf.zerosLike(weight));
    const secondMoment = state.getOrCreate("v", () => tf.zerosLike(weight));

    const nextM = tf.add(tf.mul(firstMoment, beta1), tf.mul(gradient, 1 - beta1));
    const nextV = tf.add(tf.mul(secondMoment, beta2), tf.mul(tf.square(gradient), 1 - beta2));

    // FIX: Use pure JS numeric calculations for step tracking to prevent graph memory leaks
    const mHat = tf.div(nextM, tf.scalar(1 - Math.pow(beta1, currentStep + 1)));
    const vHat = tf.div(nextV, tf.scalar(1 - Math.pow(beta2, currentStep + 1)));

    const updateDelta = tf.div(mHat, tf.add(tf.sqrt(vHat), tf.scalar(eps)));

    state.set("m", nextM);
    state.set("v", nextV);

    return updateDelta;
  }
}

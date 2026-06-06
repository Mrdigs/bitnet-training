import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "../IBitNetStrategy";
import { OptimizerState } from "../OptimizerState";

export class UncompromisedQatAdamStrategy implements IBitNetStrategy {
  private learningRate: number;
  private beta1: number;
  private beta2: number;
  private epsilon: number;

  constructor(learningRate = 0.005, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    this.learningRate = learningRate;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }

  public getPackedShape(outFeatures: number, inFeatures: number): tf.Shape {
    return [inFeatures, outFeatures];
  }

  public decodeWeights(packedTensor: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      // 1. Calculate alpha (mean of absolute weights)
      const absWeights = tf.abs(packedTensor);
      const alpha = tf.mean(absWeights);

      // 2. Prevent division by zero safely using a scalar max
      const epsilon = tf.scalar(1e-5);
      const safeAlpha = tf.maximum(alpha, epsilon);

      // 3. Normalize, clip to [-1, 1], and round to nearest integer {-1, 0, 1}
      const normalized = tf.div(packedTensor, safeAlpha);
      const clipped = tf.clipByValue(normalized, -1.0, 1.0);
      const ternarized = tf.round(clipped);

      // 4. Rescale the ternary matrix by alpha to maintain variance scale
      return tf.mul(ternarized, safeAlpha);
    });
  }

  public computeUpdate(weightVar: tf.Variable, gradient: tf.Tensor, state: OptimizerState): void {
    tf.tidy(() => {
      const timeTensor = state.get("t");
      let localT = timeTensor ? timeTensor.dataSync()[0] : 0;
      localT += 1;
      state.set("t", tf.scalar(localT));

      const shape = weightVar.shape;
      const m = state.get("m") ?? tf.zeros(shape);
      const v = state.get("v") ?? tf.zeros(shape);

      const newM = tf.add(tf.mul(m, this.beta1), tf.mul(gradient, 1 - this.beta1));
      const newV = tf.add(tf.mul(v, this.beta2), tf.mul(tf.square(gradient), 1 - this.beta2));

      state.set("m", newM);
      state.set("v", newV);

      const mHat = tf.div(newM, 1 - Math.pow(this.beta1, localT));
      const vHat = tf.div(newV, 1 - Math.pow(this.beta2, localT));

      const denominator = tf.add(tf.sqrt(vHat), this.epsilon);
      const step = tf.mul(tf.div(mHat, denominator), this.learningRate);

      weightVar.assign(tf.sub(weightVar, step));
    });
  }
}

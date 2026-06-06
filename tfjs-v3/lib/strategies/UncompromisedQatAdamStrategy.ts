import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "../IBitNetStrategy";
import { OptimizerState } from "../OptimizerState";

export class UncompromisedQatAdamStrategy implements IBitNetStrategy {
  private learningRate: number;
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private t = 0;

  constructor(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    this.learningRate = learningRate;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }

  public getPackedShape(outFeatures: number, inFeatures: number): tf.Shape {
    return [outFeatures, inFeatures];
  }

  public decodeWeights(packedTensor: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      // 1. Calculate the dynamic scale (mean of absolute tensor values)
      // This centers the thresholding box around your current weight distributions
      const scale = tf.mean(tf.abs(packedTensor));

      // 2. Add a tiny epsilon guard to prevent division-by-zero on flat states
      const safeScale = tf.add(scale, 1e-9);

      // 3. Scale, round to nearest integer, and clamp hard between -1.0 and 1.0
      // Values close to 0 will round cleanly to 0.0, creating true ternary sparsity!
      const scaled = tf.div(packedTensor, safeScale);
      const rounded = tf.round(scaled);

      return tf.clipByValue(rounded, -1.0, 1.0);
    });
  }

  public computeUpdate(weightVar: tf.Variable, gradient: tf.Tensor, state: OptimizerState): void {
    tf.tidy(() => {
      this.t += 1;

      // Clean, intuitive state lookups with simple fallbacks
      const m = state.get("m") ?? tf.zeros(weightVar.shape);
      const v = state.get("v") ?? tf.zeros(weightVar.shape);

      // Compute updates
      const newM = tf.add(tf.mul(m, this.beta1), tf.mul(gradient, 1 - this.beta1));
      const newV = tf.add(tf.mul(v, this.beta2), tf.mul(tf.square(gradient), 1 - this.beta2));

      // Overwrite values seamlessly without worrying about memory leaks
      state.set("m", newM);
      state.set("v", newV);

      // Complete the Adam step execution
      const mHat = tf.div(newM, 1 - Math.pow(this.beta1, this.t));
      const vHat = tf.div(newV, 1 - Math.pow(this.beta2, this.t));
      const denominator = tf.add(tf.sqrt(vHat), this.epsilon);
      const step = tf.mul(tf.div(mHat, denominator), this.learningRate);

      weightVar.assign(tf.sub(weightVar, step));
    });
  }
}

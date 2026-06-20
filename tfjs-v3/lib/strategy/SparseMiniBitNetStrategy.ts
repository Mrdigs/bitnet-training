import * as tf from "@tensorflow/tfjs-node";
import { PersistentState } from "../PersistentState";
import { MiniBitNetStrategy } from "./MiniBitNetStrategy";

export class SparseMiniBitNetStrategy extends MiniBitNetStrategy {
  public computeUpdate(weightTensor: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number): tf.Tensor {
    return tf.tidy(() => {
      const { weight, residual } = this.unpack(weightTensor);
      const adaptiveLearningRate = this.optmizeLearningRate(gradient, learningRate);
      // console.log(adaptiveLearningRate.dataSync().slice(0, 100));
      const updatedResidual = tf.sub(residual, tf.mul(gradient, adaptiveLearningRate));
      const updated = this.applyStochasticRounding(weight, updatedResidual);
      return this.pack(updated.ternary, updated.residual);
    });
  }

  /**
   * Gates a continuous gradient tensor based on statistical thresholding driven by the learning rate.
   *
   * @param gradient The continuous gradient tensor calculated during backpropagation.
   * @param learningRate The global learning rate, acting as the target update density (e.g., 0.05 for top 5%).
   * @returns A gated gradient tensor where only the top-k highest magnitude gradients retain their values.
   */
  public optmizeLearningRate(gradient: tf.Tensor, learningRate: number): tf.Tensor {
    return tf.tidy(() => {
      // 1. Calculate the absolute values to capture both extreme positive and negative directions
      const absGradient = tf.abs(gradient);

      // 2. Compute the Mean (μ) and Standard Deviation (σ) using fast GPU parallel reductions
      const { mean, variance } = tf.moments(absGradient);
      const stdDev = tf.sqrt(variance);

      // 3. Map the learning rate (density) to its corresponding Gaussian Z-score.
      const targetDensity = Math.min(Math.max(learningRate, 1e-5), 0.99);
      const zScore = this.getApproximateZScore(targetDensity);

      // 4. Calculate the statistical cutoff threshold (τ = μ + z*σ)
      const threshold = tf.add(mean, tf.mul(tf.scalar(zScore), stdDev));

      // 5. Generate a binary mask containing strictly 0s and 1s (1 if inside top-k, 0 if frozen)
      // tf.greater returns booleans; casting to 'float32' converts false->0 and true->1
      const binaryGatingMask = tf.cast(tf.greater(absGradient, threshold), "float32");

      return binaryGatingMask;
    });
  }

  /**
   * Approximates the Percent-Point Function (Inverse CDF) for a Standard Normal Distribution.
   * Given a target density (e.g., 0.05), it finds the z-score cut-off for a two-tailed evaluation.
   */
  /**
   * Computes the exact Percent-Point Function (Inverse CDF) for a Standard Normal Distribution.
   * Uses a highly precise rational approximation to guarantee accuracy across the extreme tail bounds.
   *
   * @param density The target update density/learning rate (e.g. 0.05)
   * @returns The precise two-tailed Z-score cut-off threshold.
   */
  public getApproximateZScore(density: number): number {
    // Convert density for a two-tailed evaluation (e.g., density 0.01 -> p = 0.995)
    const p = 1.0 - density / 2.0;

    // Coefficients for the rational approximation (Beasley-Springer-Moro / Acklam style)
    const c1 = 2.515517;
    const c2 = 0.802853;
    const c3 = 0.010328;
    const d1 = 1.432788;
    const d2 = 0.189269;
    const d3 = 0.001308;

    // We operate on the lower tail or flip for the upper tail
    const q = p > 0.5 ? 1.0 - p : p;

    // Change of variables to linearize the Gaussian tail behavior
    const t = Math.sqrt(-2.0 * Math.log(q));

    // High-precision rational polynomial fraction
    const numerator = c1 + c2 * t + c3 * t * t;
    const denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    const z = t - numerator / denominator;

    // Return the absolute positive Z-score representation
    return p > 0.5 ? z : -z;
  }
}

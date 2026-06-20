import * as tf from "@tensorflow/tfjs-node";
import { PersistentState } from "../PersistentState";
import { MiniBitNetStrategy } from "./MiniBitNetStrategy";

export class MagMiniBitNetStrategy extends MiniBitNetStrategy {
  public computeUpdate(weightTensor: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number): tf.Tensor {
    return tf.tidy(() => {
      const { weight, residual } = this.unpack(weightTensor);
      const adaptiveLearningRate = this.optmizeLearningRate(gradient, state, learningRate);
      const updatedResidual = tf.sub(residual, tf.mul(gradient, adaptiveLearningRate));
      const updated = this.applyStochasticRounding(weight, updatedResidual);
      return this.pack(updated.ternary, updated.residual);
    });
  }

  public optmizeLearningRate(gradient: tf.Tensor, state: PersistentState, learningRate: number): tf.Tensor {
    // 1. Core hyperparameters for your landscape tracking
    const alpha = 0.95; // Smoothing factor for historical magnitude (EMA)
    const epsilon = 1e-8;

    // 2. Fetch or initialize your 0-VRAM global scalar trackers
    const historicalMagnitude = state.getOrCreate("history_mag", () => tf.scalar(1.0));

    // 3. Isolate the global gradient magnitude (L1 or L2 norm) of this entire tensor
    const currentMagnitude = tf.mean(tf.abs(gradient)); // Sign-agnostic global mean magnitude

    // 4. Calculate your Landscape Ratio (Current vs History)
    const landscapeRatio = tf.div(currentMagnitude, tf.add(historicalMagnitude, tf.scalar(epsilon)));

    // 5. Meta-adjustment rule:
    // If landscapeRatio > 1, we are steepening -> shrink step size to prevent overshoot.
    // If landscapeRatio < 1, we are flattening -> boost step size to break through plateaus.
    // const adaptiveScale = tf.div(tf.scalar(1.0), tf.add(landscapeRatio, tf.scalar(epsilon)));

    // log(landscapeRatio) is perfectly symmetric.
    // We negate it because we want to multiply by a smaller number when steep (positive log)
    const sensitivity = tf.scalar(1.5); // Tune this to make it more/less aggressive
    const logRatio = tf.log(tf.add(landscapeRatio, tf.scalar(epsilon)));
    const adaptiveScale = tf.exp(tf.mul(tf.neg(logRatio), sensitivity));

    // Bound the scaling factor to prevent explosive steps or complete freezing
    const clippedScale = tf.clipByValue(adaptiveScale, 0.1, 10.0);
    const effectiveLR = tf.mul(tf.scalar(learningRate), clippedScale);

    const nextHistory = tf.add(tf.mul(historicalMagnitude, tf.scalar(alpha)), tf.mul(currentMagnitude, tf.scalar(1 - alpha)));

    // 8. Update persistent tracking states
    state.set("history_mag", nextHistory);

    return effectiveLR;
  }
}

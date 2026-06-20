import * as tf from "@tensorflow/tfjs-node";
import { ReferenceBitNetStrategy } from "./ReferenceBitNetStrategy";
import { PersistentState } from "../PersistentState";

export class MagnitudeRefStrategy extends ReferenceBitNetStrategy {
  public override computeUpdate(weight: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number, currentStep: number): tf.Tensor {
    const adaptiveLearningRate = this.optmizeLearningRate2(gradient, state, learningRate);
    const nextWeight = tf.sub(weight, tf.mul(gradient, adaptiveLearningRate));
    return nextWeight;
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
    const adaptiveScale = tf.div(tf.scalar(1.0), tf.add(landscapeRatio, tf.scalar(epsilon)));

    // Bound the scaling factor to prevent explosive steps or complete freezing
    const clippedScale = tf.clipByValue(adaptiveScale, 0.1, 10.0);
    const effectiveLR = tf.mul(tf.scalar(learningRate), clippedScale);

    const nextHistory = tf.add(tf.mul(historicalMagnitude, tf.scalar(alpha)), tf.mul(currentMagnitude, tf.scalar(1 - alpha)));

    // 8. Update persistent tracking states
    state.set("history_mag", nextHistory);

    return effectiveLR;
  }

  public optmizeLearningRate2(gradient: tf.Tensor, state: PersistentState, learningRate: number): tf.Tensor {
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

    // Modify step 5 to use a power factor (e.g., 2.0 or 3.0)
    const powerFactor = tf.scalar(2.0);

    // If ratio is 0.5 (flattening), adaptiveScale becomes (1 / 0.5)^2 = 4x boost
    // If ratio is 0.1 (severe plateau), adaptiveScale becomes (1 / 0.1)^2 = 100x boost!
    const adaptiveScale = tf.pow(tf.div(tf.scalar(1.0), tf.add(landscapeRatio, tf.scalar(epsilon))), powerFactor);

    const clippedScale = tf.clipByValue(adaptiveScale, 0.05, 50.0); // Widen maximum boost bound
    const effectiveLR = tf.mul(tf.scalar(learningRate), clippedScale);

    const nextHistory = tf.add(tf.mul(historicalMagnitude, tf.scalar(alpha)), tf.mul(currentMagnitude, tf.scalar(1 - alpha)));

    // 8. Update persistent tracking states
    state.set("history_mag", nextHistory);

    return effectiveLR;
  }
}

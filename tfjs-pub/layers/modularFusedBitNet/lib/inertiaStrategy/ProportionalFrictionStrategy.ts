/* FAILED EXPERIMENT */

import * as tf from "@tensorflow/tfjs";
import { InertiaUpdateStrategy } from "../interfaces";

export class ProportionalFrictionStrategy implements InertiaUpdateStrategy {
  private readonly frictionMultiplier: number;

  /**
   * @param frictionMultiplier Calibrates the exponential drag curve.
   * Higher values apply a heavier kinetic brake for a given gradient magnitude.
   */
  constructor(frictionMultiplier: number = 20.0) {
    this.frictionMultiplier = frictionMultiplier;
  }

  public update(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D {
    return tf.tidy(() => {
      // Explicitly clone the shared gradient input to protect the baseline memory line
      const safeGradients = gradients.clone();

      const momFloat = currentMomentum.toFloat();
      const momSign = tf.sign(momFloat);

      const gradSign = tf.neg(tf.sign(safeGradients));
      const gradMagnitude = tf.abs(safeGradients);

      // 1. Determine Directional Alignment
      const isMatch = tf.equal(gradSign, momSign);
      const isMomZero = tf.equal(momSign, tf.scalar(0.0, "float32"));
      const shouldAccelerate = tf.logicalOr(isMatch, isMomZero);
      const isGradActive = tf.notEqual(gradSign, tf.scalar(0.0, "float32"));

      const activeAcceleration = tf.logicalAnd(shouldAccelerate, isGradActive);
      const activeBraking = tf.logicalAnd(tf.logicalNot(shouldAccelerate), isGradActive);

      // 2. Stochastic Acceleration Step (Layer 1 Gate)
      const rawProb = tf.mul(gradMagnitude, tf.scalar(gradScale, "float32"));
      const accelerationProbability = tf.clipByValue(rawProb, 0.0, 1.0);
      const randSlice = tf.randomUniform(safeGradients.shape, 0.0, 1.0, "float32");
      const gatePassed = tf.less(randSlice, accelerationProbability);
      const accelStep = tf.where(gatePassed, tf.scalar(1.0, "float32"), tf.scalar(0.0, "float32"));

      // 3. CORRECTED PHYSICS: INVERTED DIVISION FLUID DECAY CURVE
      // Multiplier = 1.0 / (1.0 + (gradMagnitude * frictionMultiplier))
      const scaledMagnitude = tf.mul(gradMagnitude, tf.scalar(this.frictionMultiplier, "float32"));
      const denominator = tf.add(tf.scalar(1.0, "float32"), scaledMagnitude);
      const dynamicBrakeMultiplier = tf.div(tf.scalar(1.0, "float32"), denominator);

      // 4. Unified Branchless Blending
      const updatedMomFloat = tf.where(
        activeAcceleration,
        tf.add(momFloat, tf.mul(accelStep, gradSign)), // Standard push
        tf.where(
          activeBraking,
          tf.mul(momFloat, dynamicBrakeMultiplier), // Inverted fluid braking
          momFloat, // Zero hold
        ),
      );

      // Cast safely back to the 6-bit signed integer register block bounds (-31 to +31)
      return tf.clipByValue(tf.floor(updatedMomFloat).toInt(), -31, 31) as tf.Tensor2D;
    });
  }
}

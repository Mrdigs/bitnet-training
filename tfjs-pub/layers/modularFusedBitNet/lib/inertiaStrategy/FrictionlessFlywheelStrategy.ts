import * as tf from "@tensorflow/tfjs";
import { InertiaUpdateStrategy } from "../interfaces";

/**
 * FrictionlessFlywheelStrategy (Iteration 2.6 Core)
 * Implements a pure cumulative momentum engine matching Newton's First Law.
 * Completely removes global friction decay constants. Momentum updates are 100%
 * cumulative, allowing the 1.000 probability ceiling at indices 26-31 of our
 * right-side-up LUT to act as the native, organic inertial shield.
 * Old velocity is only drained by active, opposing backprop gradient work.
 */
export class FrictionlessFlywheelStrategy implements InertiaUpdateStrategy {
  private readonly frictionCoeff: tf.Scalar;

  constructor() {
    // Enforce absolute zero global friction drag (lambda = 1.000)
    // to let the look-up table handle 100% of parameter stabilization.
    this.frictionCoeff = tf.scalar(1.0, "float32");
  }

  public update(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D {
    return tf.tidy(() => {
      const safeGradients = gradients.clone();
      const momFloat = currentMomentum.toFloat();
      const scaleForce = tf.scalar(gradScale, "float32");

      // Line 1: Pure branchless vector momentum accumulation (Frictionless Flywheel)
      const rawNextMom = tf.sub(tf.mul(momFloat, this.frictionCoeff), tf.mul(safeGradients, scaleForce)) as tf.Tensor2D;

      // Strict 6-bit signed integer clipping protection boundaries (-31 to +31)
      const boundedMom = tf.clipByValue(rawNextMom, -31.0, 31.0);

      // Line 2: Stochastic Floor Rounding Layer 1 Gate (Casts back to integer space)
      const floorMom = tf.floor(boundedMom);
      const fractionalPart = tf.sub(boundedMom, floorMom);
      const randSlice = tf.randomUniform(gradients.shape, 0.0, 1.0, "float32");
      const roundUp = tf.less(randSlice, fractionalPart);

      return tf.where(roundUp, tf.ceil(boundedMom), floorMom).toInt() as tf.Tensor2D;
    });
  }
}

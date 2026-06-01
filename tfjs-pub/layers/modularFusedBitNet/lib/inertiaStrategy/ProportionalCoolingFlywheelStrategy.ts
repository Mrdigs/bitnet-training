import * as tf from "@tensorflow/tfjs";
import { InertiaUpdateStrategy } from "../interfaces";

/**
 * ProportionalCoolingFlywheelStrategy (Iteration 2.7 Calibrated)
 * Implements a Proportional Thermodynamic Friction Model.
 * Dynamically scales the time decay friction lambda based on gradient magnitude.
 * Active gradients unlock an un-throttled frictionless flywheel (lambda -> 1.00),
 * while collapsing gradients automatically trigger high-mass cooling (lambda -> 0.75)
 * to bleed off momentum and settle the parameter safely into the rest cradle.
 */
export class ProportionalCoolingFlywheelStrategy implements InertiaUpdateStrategy {
  // FIX: Store the bounding configuration values as native JS primitive numbers
  // to perfectly satisfy the strict tf.clipByValue primitive signature parameters!
  private readonly baseFriction: number;
  private readonly maxFriction: number;
  private readonly sensitivity: number;

  constructor(baseFriction: number = 0.75, maxFriction: number = 1.0, sensitivity: number = 1.5) {
    this.baseFriction = baseFriction;
    this.maxFriction = maxFriction;
    this.sensitivity = sensitivity;
  }

  public update(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D {
    return tf.tidy(() => {
      const safeGradients = gradients.clone();
      const momFloat = currentMomentum.toFloat();
      const scaleForce = tf.scalar(gradScale, "float32");

      // 1. Calculate Proportional Friction Lambda Matrix Element-Wise
      // lambda = clip(0.75 + |grad| * sensitivity, 0.75, 1.00)
      const gradMagnitude = tf.abs(safeGradients);
      const baseFrictionScalar = tf.scalar(this.baseFriction, "float32");
      const sensitivityScalar = tf.scalar(this.sensitivity, "float32");

      const rawLambda = tf.add(baseFrictionScalar, tf.mul(gradMagnitude, sensitivityScalar));

      // Fix applied: Passing raw numbers natively into the bounding clipping function
      const dynamicLambda = tf.clipByValue(rawLambda, this.baseFriction, this.maxFriction);

      // 2. Pure Branchless Two-Line Warp Calculus Core
      const rawNextMom = tf.sub(tf.mul(momFloat, dynamicLambda), tf.mul(safeGradients, scaleForce)) as tf.Tensor2D;

      // Strict 6-bit signed integer register clipping protection boundaries (-31 to +31)
      const boundedMom = tf.clipByValue(rawNextMom, -31.0, 31.0);

      // 3. Stochastic Floor Rounding Layer 1 Gate
      const floorMom = tf.floor(boundedMom);
      const fractionalPart = tf.sub(boundedMom, floorMom);
      const randSlice = tf.randomUniform(gradients.shape, 0.0, 1.0, "float32");
      const roundUp = tf.less(randSlice, fractionalPart);

      return tf.where(roundUp, tf.ceil(boundedMom), floorMom).toInt() as tf.Tensor2D;
    });
  }
}

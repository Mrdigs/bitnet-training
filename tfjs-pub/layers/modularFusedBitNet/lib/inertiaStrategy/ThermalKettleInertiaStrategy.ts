import * as tf from "@tensorflow/tfjs";
import { InertiaUpdateStrategy } from "../interfaces";

/**
 * ThermalKettleInertiaStrategy (Iteration 2.3 Core)
 * Implements a Dynamic Thermal Friction Model.
 * Natively evaluates directional consensus: if gradients match momentum history
 * (boiling), time friction drops to 0.0 (lambda = 1.0). If gradients oppose or die,
 * friction slams to maximum (lambda = 0.70) to instantly drain old inertia memory.
 */
export class ThermalKettleInertiaStrategy implements InertiaUpdateStrategy {
  private readonly boilingLambda: tf.Scalar;
  private readonly coolingLambda: tf.Scalar;

  constructor(boilingLambda: number = 1.0, coolingLambda: number = 0.7) {
    this.boilingLambda = tf.scalar(boilingLambda, "float32");
    this.coolingLambda = tf.scalar(coolingLambda, "float32");
  }

  public update(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D {
    return tf.tidy(() => {
      const safeGradients = gradients.clone();
      const momFloat = currentMomentum.toFloat();
      const scaleForce = tf.scalar(gradScale, "float32");

      // 1. Evaluate Directional Alignment (Boiling Consensus Map)
      const momSign = tf.sign(momFloat);
      const gradSign = tf.neg(tf.sign(safeGradients)); // Negative gradient pushes positive

      // True if incoming energy actively matches historical directional trajectory
      const isBoiling = tf.equal(momSign, gradSign);
      const isMomZero = tf.equal(momSign, tf.scalar(0.0, "float32"));
      const activeConsensus = tf.logicalOr(isBoiling, isMomZero);

      // 2. Dynamic Matrix Friction Assignment
      // Natively selects lambda per parameter based on structural consensus
      const dynamicLambda = tf.where(
        activeConsensus,
        this.boilingLambda, // Boiling: Zero friction hold (1.00)
        this.coolingLambda, // Cooling: Instant energy drain (0.70)
      );

      // 3. The Two-Line Warp Calculations
      const rawNextMom = tf.sub(tf.mul(momFloat, dynamicLambda), tf.mul(safeGradients, scaleForce)) as tf.Tensor2D;

      const boundedMom = tf.clipByValue(rawNextMom, -32.0, 31.0);

      // 4. Stochastic Floor Write-Back Gate
      const floorMom = tf.floor(boundedMom);
      const fractionalPart = tf.sub(boundedMom, floorMom);
      const randSlice = tf.randomUniform(gradients.shape, 0.0, 1.0, "float32");
      const roundUp = tf.less(randSlice, fractionalPart);

      return tf.where(roundUp, tf.ceil(boundedMom), floorMom).toInt() as tf.Tensor2D;
    });
  }
}

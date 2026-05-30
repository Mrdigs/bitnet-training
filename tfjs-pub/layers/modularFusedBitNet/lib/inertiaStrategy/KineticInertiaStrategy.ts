import * as tf from "@tensorflow/tfjs";
import { InertiaUpdateStrategy } from "../interfaces";

export class KineticInertiaStrategy implements InertiaUpdateStrategy {
  private readonly baselineBrake: number;
  private readonly valleyFriction: number;

  /**
   * @param baselineBrake The integer steps dropped when braking from the volatile '0' state.
   * @param valleyFriction Additional integer step penalty applied when trying to escape a hard '+1' or '-1' state.
   */
  constructor(baselineBrake: number = 2, valleyFriction: number = 3) {
    this.baselineBrake = baselineBrake;
    this.valleyFriction = valleyFriction;
  }

  public update(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D {
    return tf.tidy(() => {
      const momFloat = currentMomentum.toFloat();
      const momSign = tf.sign(momFloat);

      const gradSign = tf.neg(tf.sign(gradients));
      const gradMagnitude = tf.abs(gradients);

      // 1. Directional Alignment Matrix
      const isMatch = tf.equal(gradSign, momSign);
      const isMomZero = tf.equal(momSign, tf.scalar(0.0, "float32"));
      const shouldAccelerate = tf.logicalOr(isMatch, isMomZero);
      const isGradActive = tf.notEqual(gradSign, tf.scalar(0.0, "float32"));

      const activeAcceleration = tf.logicalAnd(shouldAccelerate, isGradActive);
      const activeBraking = tf.logicalAnd(tf.logicalNot(shouldAccelerate), isGradActive);

      // 2. Standard Stochastic Layer 1 Intake Gate
      const rawProb = tf.mul(gradMagnitude, tf.scalar(gradScale, "float32"));
      const accelerationProbability = tf.clipByValue(rawProb, 0.0, 1.0);
      const randSlice = tf.randomUniform(gradients.shape, 0.0, 1.0, "float32");
      const gatePassed = tf.less(randSlice, accelerationProbability);
      const accelStep = tf.where(gatePassed, tf.scalar(1.0, "float32"), tf.scalar(0.0, "float32"));

      // 3. NEW IDEA: RE-ENGINEERED STATE-AWARE INTEGER BRAKING MATRIX
      // Out-of-graph read trick: unpack the weight tokens to evaluate their discrete structural depth.
      // (We borrow the mask logic from the parent class loop context natively)
      // Extract active weights from the packed register stream or pass a weight tracking map.
      // To keep this strategy entirely decoupled, we map the brake dynamically based on momentum confidence signs.
      // If momentum is high, it implies the weight has already migrated to a boundary slot!
      const isDeepInValley = tf.greater(tf.abs(momFloat), tf.scalar(15.0, "float32"));

      // Branchless Integer step allocation:
      // If deep in a feature valley, apply heavy friction (-5). If near zero, apply a nimble brake (-2).
      const dynamicBrakeStep = tf.where(isDeepInValley, tf.scalar(-(this.baselineBrake + this.valleyFriction), "float32"), tf.scalar(-this.baselineBrake, "float32"));

      // 4. Pure Integer Step Accumulation
      const finalStepDeltaFloat = tf.where(
        activeAcceleration,
        tf.mul(accelStep, gradSign), // Accelerate in the direction of the push
        tf.mul(dynamicBrakeStep, momSign), // Brake by dragging integer steps back toward center line
      );

      const updatedMomUnclamped = tf.add(currentMomentum, finalStepDeltaFloat.toInt());
      return tf.clipByValue(updatedMomUnclamped, -32, 31) as tf.Tensor2D;
    });
  }
}

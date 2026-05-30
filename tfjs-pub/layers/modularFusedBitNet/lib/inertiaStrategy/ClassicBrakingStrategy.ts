import { InertiaUpdateStrategy } from "../interfaces";
import * as tf from "@tensorflow/tfjs";

// --- The Original Rigid-Braking Momentum Update ---
export class ClassicBrakingStrategy implements InertiaUpdateStrategy {
  public update(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D {
    return tf.tidy(() => {
      const momFloat = currentMomentum.toFloat();
      const momSign = tf.sign(momFloat);
      const gradSign = tf.neg(tf.sign(gradients));
      const gradMagnitude = tf.abs(gradients);

      const isMatch = tf.equal(gradSign, momSign);
      const isMomZero = tf.equal(momSign, tf.scalar(0.0, "float32"));
      const shouldAccelerate = tf.logicalOr(isMatch, isMomZero);
      const isGradActive = tf.notEqual(gradSign, tf.scalar(0.0, "float32"));

      const activeAcceleration = tf.logicalAnd(shouldAccelerate, isGradActive);
      const activeBraking = tf.logicalAnd(tf.logicalNot(shouldAccelerate), isGradActive);

      const rawProb = tf.mul(gradMagnitude, tf.scalar(gradScale, "float32"));
      const accelerationProbability = tf.clipByValue(rawProb, 0.0, 1.0);
      const randSlice = tf.randomUniform(gradients.shape, 0.0, 1.0, "float32");
      const gatePassed = tf.less(randSlice, accelerationProbability);

      const accelStep = tf.where(gatePassed, tf.scalar(1.0, "float32"), tf.scalar(0.0, "float32"));
      const brakeStep = tf.scalar(-3.0, "float32"); // Our classic rigid linear brake

      const finalStepDeltaFloat = tf.where(activeAcceleration, tf.mul(accelStep, gradSign), tf.mul(brakeStep, momSign));

      const updatedMomUnclamped = tf.add(currentMomentum, finalStepDeltaFloat.toInt());
      return tf.clipByValue(updatedMomUnclamped, -32, 31) as tf.Tensor2D;
    });
  }
}

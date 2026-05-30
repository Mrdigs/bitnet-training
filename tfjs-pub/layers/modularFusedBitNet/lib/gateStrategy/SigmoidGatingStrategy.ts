import * as tf from "@tensorflow/tfjs";
import { StochasticGateStrategy } from "../interfaces";

/**
 * SigmoidGatingStrategy (Iteration 2.8 Calibrated)
 * Replaces the rigid static 33-element LUT array with a continuous,
 * vectorized Logistic Sigmoid function. Ensures that lower-velocity registers
 * maintain a healthy exploratory baseline probability (5-10%) to un-choke
 * micro-gradient training passes, optimizing perfectly for GPU warps.
 */
export class SigmoidGatingStrategy implements StochasticGateStrategy {
  private readonly alpha: tf.Scalar; // Center shift offset threshold
  private readonly beta: tf.Scalar; // Temperature fluidity divisor

  constructor(alpha: number = 8.0, beta: number = 3.0) {
    this.alpha = tf.scalar(alpha, "float32");
    this.beta = tf.scalar(beta, "float32");
  }

  public evaluate(weight: tf.Tensor2D, momentum: tf.Tensor2D, scale_t: number): { updatedWeight: tf.Tensor2D; dampedMomentum: tf.Tensor2D } {
    return tf.tidy(() => {
      const absVelocity = tf.abs(momentum).toFloat();

      // Scaled continuous thermal activation equation:
      // logits = (|momentum| - alpha) / (beta * scale_t)
      const scaledBeta = tf.mul(this.beta, tf.scalar(scale_t, "float32"));
      const logits = tf.div(tf.sub(absVelocity, this.alpha), scaledBeta);
      const flipProbability = tf.sigmoid(logits);

      const randSlice = tf.randomUniform(momentum.shape, 0.0, 1.0, "float32");
      const shouldFlip = tf.less(randSlice, flipProbability);

      const isMomPositive = tf.greater(momentum, tf.scalar(0, "int32"));
      const isMomNegative = tf.less(momentum, tf.scalar(0, "int32"));

      const isAtMinusOne = tf.equal(weight, tf.scalar(0, "int32"));
      const isAtZero = tf.equal(weight, tf.scalar(1, "int32"));
      const isAtPlusOne = tf.equal(weight, tf.scalar(2, "int32"));

      const canIncrement = tf.logicalOr(isAtMinusOne, isAtZero);
      const canDecrement = tf.logicalOr(isAtPlusOne, isAtZero);

      const weightIncrement = tf.logicalAnd(tf.logicalAnd(shouldFlip, isMomPositive), canIncrement).toInt();
      const weightDecrement = tf.logicalAnd(tf.logicalAnd(shouldFlip, isMomNegative), canDecrement).toInt();

      const updatedWeight = tf.add(tf.sub(weight, weightDecrement), weightIncrement) as tf.Tensor2D;

      // FIX: Enforce perfect structural rank-2 matching by utilizing tf.zerosLike()
      // instead of a 0D tf.scalar to completely eliminate the ts(2352) compilation mismatch!
      const actualFlipOccurred = tf.logicalOr(tf.greater(weightIncrement, 0), tf.greater(weightDecrement, 0));
      const zeroResetBuffer = tf.zerosLike(momentum);
      const dampedMomentum = tf.where(actualFlipOccurred, zeroResetBuffer, momentum) as tf.Tensor2D;

      return { updatedWeight, dampedMomentum };
    });
  }
}

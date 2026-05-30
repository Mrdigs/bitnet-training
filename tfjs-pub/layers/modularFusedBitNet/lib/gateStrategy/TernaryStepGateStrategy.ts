// layers/modularFusedBitNet/lib/gateStrategy/TernaryStepGateStrategy.ts
import * as tf from "@tensorflow/tfjs";
import { StochasticGateStrategy } from "../interfaces";

export class TernaryStepGateStrategy implements StochasticGateStrategy {
  private static readonly LUT_DATA = [0.0, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.044, 0.055, 0.069, 0.086, 0.107, 0.134, 0.168, 0.21, 0.262, 0.328, 0.41, 0.512, 0.64, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
  private static readonly LUT_TENSOR = tf.keep(tf.tensor1d(TernaryStepGateStrategy.LUT_DATA, "float32"));

  public evaluate(weight: tf.Tensor2D, momentum: tf.Tensor2D, scale_t: number): { updatedWeight: tf.Tensor2D; dampedMomentum: tf.Tensor2D } {
    return tf.tidy(() => {
      const absVelocity = tf.abs(momentum);
      const annealedIndex = tf.clipByValue(tf.mul(absVelocity.toFloat(), tf.scalar(scale_t, "float32")).toInt(), 0, 32);
      const flipProbability = tf.gather(TernaryStepGateStrategy.LUT_TENSOR, annealedIndex);

      const randSlice = tf.randomUniform(momentum.shape, 0.0, 1.0, "float32");
      const shouldFlip = tf.less(randSlice, flipProbability);

      const isMomPositive = tf.greater(momentum, tf.scalar(0, "int32"));
      const isMomNegative = tf.less(momentum, tf.scalar(0, "int32"));

      // ZERO-CROSSING SAFETY FILTER:
      // If momentum pushes positive, we only increment if weight is strictly less than neutrality (token 1)
      // OR if weight is exactly at neutrality (token 1) to transition to +1 (token 2).
      // This physically blocks a token at 0 (-1) from stepping straight to 2 (+1) in a single pass!
      const weightFloat = weight.toFloat();
      const isAtMinusOne = tf.equal(weight, tf.scalar(0, "int32"));
      const isAtZero = tf.equal(weight, tf.scalar(1, "int32"));
      const isAtPlusOne = tf.equal(weight, tf.scalar(2, "int32"));

      const canIncrement = tf.logicalOr(isAtMinusOne, isAtZero);
      const canDecrement = tf.logicalOr(isAtPlusOne, isAtZero);

      const weightIncrement = tf.logicalAnd(tf.logicalAnd(shouldFlip, isMomPositive), canIncrement).toInt();
      const weightDecrement = tf.logicalAnd(tf.logicalAnd(shouldFlip, isMomNegative), canDecrement).toInt();

      const updatedWeight = tf.add(tf.sub(weight, weightDecrement), weightIncrement) as tf.Tensor2D;

      // Apply a clean 50% velocity damping wash-out on parameters that execute a transition step
      const actualFlipOccurred = tf.logicalOr(tf.greater(weightIncrement, 0), tf.greater(weightDecrement, 0));
      const dampedMomentum = tf.where(actualFlipOccurred, tf.mul(momentum, 0.5).toInt(), momentum) as tf.Tensor2D;

      return { updatedWeight, dampedMomentum };
    });
  }
}

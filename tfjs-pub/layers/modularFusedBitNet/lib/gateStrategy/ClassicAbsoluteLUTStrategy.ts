import { StochasticGateStrategy } from "../interfaces";
import * as tf from "@tensorflow/tfjs";

// --- The Original Base-1.25 Absolute LUT Update ---
export class ClassicAbsoluteLUTStrategy implements StochasticGateStrategy {
  // This is so wrong. The second line contains massive performance gains, but this is now in ThresholdAccumulatorLUTStrategy
  private static readonly LOG_PROB_LUT_DATA = [1.0, 0.8, 0.64, 0.512, 0.41, 0.328, 0.262, 0.21, 0.168, 0.134, 0.107, 0.086, 0.069, 0.055, 0.044, 0.035, 0.028, 0.023, 0.018, 0.014, 0.012, 0.009, 0.007, 0.006, 0.005, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  // private static readonly LOG_PROB_LUT_DATA = [0.0, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.044, 0.055, 0.069, 0.086, 0.107, 0.134, 0.168, 0.21, 0.262, 0.328, 0.41, 0.512, 0.64, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
  private static readonly LUT_TENSOR = tf.keep(tf.tensor1d(ClassicAbsoluteLUTStrategy.LOG_PROB_LUT_DATA, "float32"));

  public evaluate(weight: tf.Tensor2D, momentum: tf.Tensor2D, scale_t: number): { updatedWeight: tf.Tensor2D; dampedMomentum: tf.Tensor2D } {
    return tf.tidy(() => {
      const absVelocity = tf.abs(momentum);
      const annealedIndexFloat = tf.mul(absVelocity.toFloat(), tf.scalar(scale_t, "float32"));
      const annealedIndex = tf.clipByValue(annealedIndexFloat.toInt(), 0, 32);

      const flipProbability = tf.gather(ClassicAbsoluteLUTStrategy.LUT_TENSOR, annealedIndex);
      const randSlice2 = tf.randomUniform(momentum.shape, 0.0, 1.0, "float32");
      const shouldFlip = tf.less(randSlice2, flipProbability);

      const isMomPositive = tf.greater(momentum, tf.scalar(0, "int32"));
      const isMomNegative = tf.less(momentum, tf.scalar(0, "int32"));
      const triggerUp = tf.logicalAnd(shouldFlip, isMomPositive);
      const triggerDown = tf.logicalAnd(shouldFlip, isMomNegative);

      const isNotMax = tf.notEqual(weight, tf.scalar(2, "int32"));
      const weightIncrement = tf.logicalAnd(triggerUp, isNotMax).toInt();

      const isNotMin = tf.notEqual(weight, tf.scalar(0, "int32"));
      const weightDecrement = tf.logicalAnd(triggerDown, isNotMin).toInt();

      const updatedWeight = tf.add(tf.sub(weight, weightDecrement), weightIncrement) as tf.Tensor2D;

      const actualFlipOccurred = tf.logicalOr(tf.greater(weightIncrement, 0), tf.greater(weightDecrement, 0));
      const positiveDamp = tf.clipByValue(tf.sub(momentum, tf.scalar(3, "int32")), 0, 31);
      const negativeDamp = tf.clipByValue(tf.add(momentum, tf.scalar(3, "int32")), -32, 0);

      const dampedState = tf.where(isMomPositive, positiveDamp, negativeDamp);
      const dampedMomentum = tf.where(actualFlipOccurred, dampedState, momentum) as tf.Tensor2D;

      return { updatedWeight, dampedMomentum };
    });
  }
}

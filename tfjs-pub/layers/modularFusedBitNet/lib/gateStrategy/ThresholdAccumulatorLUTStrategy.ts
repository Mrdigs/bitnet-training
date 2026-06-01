import * as tf from "@tensorflow/tfjs";
import { StochasticGateStrategy } from "../interfaces";

export class ThresholdAccumulatorLUTStrategy implements StochasticGateStrategy {
  // --- THE ACCUMULATOR LUT BLUEPRINT ---
  // Mathematically inverted to enforce Critical Mass Threshold Mechanics.
  // Index 0 (Momentum 0) = 0% volatility. Index 26-31 = 100% phase transition probability.
  private static readonly ACCUMULATOR_LUT_DATA = [0.0, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.044, 0.055, 0.069, 0.086, 0.107, 0.134, 0.168, 0.21, 0.262, 0.328, 0.41, 0.512, 0.64, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
  private static readonly LUT_TENSOR = tf.keep(tf.tensor1d(ThresholdAccumulatorLUTStrategy.ACCUMULATOR_LUT_DATA, "float32"));

  public evaluate(weight: tf.Tensor2D, momentum: tf.Tensor2D, scale_t: number): { updatedWeight: tf.Tensor2D; dampedMomentum: tf.Tensor2D } {
    return tf.tidy(() => {
      const absVelocity = tf.abs(momentum);

      // 1. Calculate Annealed Lookup Index Natively
      // As scale_t inflates late in training via host annealing, it pushes
      // active values deeper into the high-index thresholds, hardening the updates.
      const annealedIndexFloat = tf.mul(absVelocity.toFloat(), tf.scalar(scale_t, "float32"));
      const annealedIndex = tf.clipByValue(annealedIndexFloat.toInt(), 0, 32);

      // 2. Vectorized Register LUT Fetch
      const flipProbability = tf.gather(ThresholdAccumulatorLUTStrategy.LUT_TENSOR, annealedIndex);

      // 3. Stochastic Rounding Layer 2 (The Flip Gate)
      const randSlice2 = tf.randomUniform(momentum.shape, 0.0, 1.0, "float32");
      const shouldFlip = tf.less(randSlice2, flipProbability);

      // 4. Directional Weight Transitions
      const isMomPositive = tf.greater(momentum, tf.scalar(0, "int32"));
      const isMomNegative = tf.less(momentum, tf.scalar(0, "int32"));
      const triggerUp = tf.logicalAnd(shouldFlip, isMomPositive);
      const triggerDown = tf.logicalAnd(shouldFlip, isMomNegative);

      // Enforce ternary ceiling boundary constraints (+1 is stored as token value 2)
      const isNotMax = tf.notEqual(weight, tf.scalar(2, "int32"));
      const weightIncrement = tf.logicalAnd(triggerUp, isNotMax).toInt();

      // Enforce ternary floor boundary constraints (-1 is stored as token value 0)
      const isNotMin = tf.notEqual(weight, tf.scalar(0, "int32"));
      const weightDecrement = tf.logicalAnd(triggerDown, isNotMin).toInt();

      // Execute zero-pass state updates branchlessly
      const updatedWeight = tf.add(tf.sub(weight, weightDecrement), weightIncrement) as tf.Tensor2D;

      // 5. Base-1.25 Logarithmic Velocity Damper
      // If a parameter accumulates critical mass and successfully executes a flip,
      // it cannot remain at maximum volatility or it will immediately overshoot on the next batch.
      // We step the counter back toward zero by 3 units to damp its immediate velocity.
      const actualFlipOccurred = tf.logicalOr(tf.greater(weightIncrement, 0), tf.greater(weightDecrement, 0));
      const positiveDamp = tf.clipByValue(tf.sub(momentum, tf.scalar(3, "int32")), 0, 31);
      const negativeDamp = tf.clipByValue(tf.add(momentum, tf.scalar(3, "int32")), -31, 0);

      const dampedState = tf.where(isMomPositive, positiveDamp, negativeDamp);
      const dampedMomentum = tf.where(actualFlipOccurred, dampedState, momentum) as tf.Tensor2D;

      return { updatedWeight, dampedMomentum };
    });
  }
}

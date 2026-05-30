import * as tf from "@tensorflow/tfjs";
import { StochasticGateStrategy } from "../interfaces";

export class DirectionAwareLUTStrategy implements StochasticGateStrategy {
  // Standard Base-1.25 Look-Up Table for base tracking
  private static readonly BASE_LUT_DATA = [1.0, 0.8, 0.64, 0.512, 0.41, 0.328, 0.262, 0.21, 0.168, 0.134, 0.107, 0.086, 0.069, 0.055, 0.044, 0.035, 0.028, 0.023, 0.018, 0.014, 0.012, 0.009, 0.007, 0.006, 0.005, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  private static readonly LUT_TENSOR = tf.keep(tf.tensor1d(DirectionAwareLUTStrategy.BASE_LUT_DATA, "float32"));

  public evaluate(weight: tf.Tensor2D, momentum: tf.Tensor2D, scale_t: number): { updatedWeight: tf.Tensor2D; dampedMomentum: tf.Tensor2D } {
    return tf.tidy(() => {
      const momFloat = momentum.toFloat();
      const absVelocity = tf.abs(momentum);

      // 1. Calculate Annealed Lookup Index Natively
      const annealedIndexFloat = tf.mul(absVelocity.toFloat(), tf.scalar(scale_t, "float32"));
      const annealedIndex = tf.clipByValue(annealedIndexFloat.toInt(), 0, 32);

      // Fetch standard baseline probabilities from our pre-allocated tensor
      const baseProbability = tf.gather(DirectionAwareLUTStrategy.LUT_TENSOR, annealedIndex);

      // 2. FEATURE FLUIDITY: Direction-Aware Velocity Adjustment
      // We check if the parameter weight sign and momentum sign are pointing in the same direction.
      // If they match (and aren't zero), the parameter is actively accelerating AWAY from the baseline.
      const weightSign = tf.sign(tf.sub(weight.toFloat(), tf.scalar(1.0, "float32")));
      const momentumSign = tf.sign(momFloat);

      // True if accelerating out into feature space, false if dragging back to zero or resting
      const isAcceleratingAway = tf.logicalAnd(tf.equal(weightSign, momentumSign), tf.notEqual(weightSign, tf.scalar(0.0, "float32")));

      // Branchless Probability Boosting:
      // If accelerating away, inflate the flip probability threshold by a factor of 1.2x
      // to smooth out the transition out of the absolute zero bog.
      const adjustedProbability = tf.where(isAcceleratingAway, tf.clipByValue(tf.mul(baseProbability, tf.scalar(1.2, "float32")), 0.0, 1.0), baseProbability);

      // 3. Execute Stochastic Flip Pass
      const randSlice2 = tf.randomUniform(momentum.shape, 0.0, 1.0, "float32");
      const shouldFlip = tf.less(randSlice2, adjustedProbability);

      const isMomPositive = tf.greater(momentum, tf.scalar(0, "int32"));
      const isMomNegative = tf.less(momentum, tf.scalar(0, "int32"));
      const triggerUp = tf.logicalAnd(shouldFlip, isMomPositive);
      const triggerDown = tf.logicalAnd(shouldFlip, isMomNegative);

      const isNotMax = tf.notEqual(weight, tf.scalar(2, "int32"));
      const weightIncrement = tf.logicalAnd(triggerUp, isNotMax).toInt();

      const isNotMin = tf.notEqual(weight, tf.scalar(0, "int32"));
      const weightDecrement = tf.logicalAnd(triggerDown, isNotMin).toInt();

      const updatedWeight = tf.add(tf.sub(weight, weightDecrement), weightIncrement) as tf.Tensor2D;

      // 4. Logarithmic Velocity Damper (Stepping back by 3 linear units if flipped)
      const actualFlipOccurred = tf.logicalOr(tf.greater(weightIncrement, 0), tf.greater(weightDecrement, 0));
      const positiveDamp = tf.clipByValue(tf.sub(momentum, tf.scalar(3, "int32")), 0, 31);
      const negativeDamp = tf.clipByValue(tf.add(momentum, tf.scalar(3, "int32")), -32, 0);

      const dampedState = tf.where(isMomPositive, positiveDamp, negativeDamp);
      const dampedMomentum = tf.where(actualFlipOccurred, dampedState, momentum) as tf.Tensor2D;

      return { updatedWeight, dampedMomentum };
    });
  }
}

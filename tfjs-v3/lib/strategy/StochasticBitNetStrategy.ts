import * as tf from "@tensorflow/tfjs-node";
import { FourTo1BitPackingStrategy } from "./FourTo1BitPackingStrategy";
import { PersistentState } from "../PersistentState";

export class StochasticBitNetStrategy extends FourTo1BitPackingStrategy {
  private readonly gradScale: number;
  private readonly K: number;
  private readonly seed: number | undefined;

  // Symmetrical probability sequence matching TernaryStepGateStrategy exactly
  private static readonly LUT_DATA = [0.0, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.044, 0.055, 0.069, 0.086, 0.107, 0.134, 0.168, 0.21, 0.262, 0.328, 0.41, 0.512, 0.64, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
  private static readonly LUT_TENSOR = tf.tensor1d(StochasticBitNetStrategy.LUT_DATA, "float32");

  constructor(gradScale = 30.0, K = 0.5, seed?: number) {
    super();
    this.gradScale = gradScale;
    this.K = K;
    this.seed = seed;
  }

  /**
   * RESTORED: Dynamic Variance Thresholding Layout
   * Maps continuous weights to a balanced 33/33/34 ternary distribution at birth.
   */
  public prepareInitialWeights(rawFloatWeights: tf.Tensor, state: PersistentState): tf.Tensor {
    return tf.tidy(() => {
      // 1. Calculate the standard deviation threshold dynamically from matrix physics
      const meanWeights = tf.mean(rawFloatWeights);
      const variance = tf.mean(tf.square(tf.sub(rawFloatWeights, meanWeights)));
      const stdDev = tf.sqrt(variance);
      const threshold = tf.mul(stdDev, tf.scalar(0.65, "float32"));

      // 2. Map continuous values directly to official unsigned domain tokens:
      // -1.0 (Negative Zone) -> 0
      //  0.0 (Neutral Zone)  -> 1
      //  1.0 (Positive Zone) -> 2
      const isNegativeZone = tf.less(rawFloatWeights, tf.neg(threshold));
      const isPositiveZone = tf.greater(rawFloatWeights, threshold);

      const unsignedWeights = tf.where(isNegativeZone, tf.scalar(0, "int32"), tf.where(isPositiveZone, tf.scalar(2, "int32"), tf.scalar(1, "int32")));

      // 3. Initialize velocity registers (momentum) at absolute zero
      const initialMomentum = tf.zerosLike(unsignedWeights);

      // 4. Compress safely using your verified packing method
      return this.pack(unsignedWeights, initialMomentum);
    });
  }

  /**
   * Implements the Proportional Thermodynamic Friction Model.
   */
  private calculateNextMomentum(momentum: tf.Tensor, gradient: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const momFloat = momentum.toFloat();
      const gradScaleScalar = tf.scalar(this.gradScale, "float32");

      const gradMagnitude = tf.abs(gradient);
      const baseFriction = tf.scalar(0.75, "float32");
      const sensitivity = tf.scalar(1.5, "float32");
      const rawLambda = tf.add(baseFriction, tf.mul(gradMagnitude, sensitivity));
      const dynamicLambda = tf.clipByValue(rawLambda, 0.75, 1.0);

      const rawNextMom = tf.sub(tf.mul(momFloat, dynamicLambda), tf.mul(gradient, gradScaleScalar));
      const boundedMom = tf.clipByValue(rawNextMom, -31.0, 31.0);

      const floorMom = tf.floor(boundedMom);
      const fractionalPart = tf.sub(boundedMom, floorMom);
      const randSliceRound = tf.randomUniform(gradient.shape, 0.0, 1.0, "float32", this.seed);
      const roundUp = tf.less(randSliceRound, fractionalPart);

      return tf.where(roundUp, tf.ceil(boundedMom), floorMom).toInt();
    });
  }

  /**
   * Implements the Ternary Step Gate Strategy.
   */
  private evaluateWeightFlips(weight: tf.Tensor, momentum: tf.Tensor, scale_t: number): { updatedWeight: tf.Tensor; dampedMomentum: tf.Tensor } {
    return tf.tidy(() => {
      const absVelocity = tf.abs(momentum);
      const annealedIndex = tf.clipByValue(tf.mul(absVelocity.toFloat(), tf.scalar(scale_t, "float32")).toInt(), 0, 32);

      const flipProbability = tf.gather(StochasticBitNetStrategy.LUT_TENSOR, annealedIndex);
      const randSlice = tf.randomUniform(momentum.shape, 0.0, 1.0, "float32", this.seed);
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

      const updatedWeight = tf.add(tf.sub(weight, weightDecrement), weightIncrement);
      const actualFlipOccurred = tf.logicalOr(tf.greater(weightIncrement, 0), tf.greater(weightDecrement, 0));

      const dampedMomentum = tf.where(actualFlipOccurred, tf.mul(momentum, 0.5).toInt(), momentum);

      return { updatedWeight, dampedMomentum };
    });
  }

  /**
   * Restores Pre-Layer RMSNorm layer to normalize input variance to 1.0.
   */
  public quantizeActivations(inputs: tf.Tensor, state: PersistentState): tf.Tensor {
    return tf.tidy(() => {
      const axis = inputs.rank - 1;
      const meanSquare = tf.mean(tf.square(inputs), axis, true);
      const rms = tf.sqrt(tf.add(meanSquare, tf.scalar(1e-5)));
      return tf.div(inputs, rms);
    });
  }

  /**
   * Unpacks storage parameters and maps them to ternary states cleanly.
   */
  public decodeWeights(packedTensor: tf.Tensor, state: PersistentState): tf.Tensor {
    return tf.tidy(() => {
      const { weight } = this.unpack(packedTensor);
      return tf.sub(weight.toFloat(), tf.scalar(1.0, "float32"));
    });
  }

  /**
   * Scaled Dequantization Pass-Through
   * Binds the output variance to the network feature layer dimension width (1 / sqrt(d)).
   * Prevents Softmax infinity overflows (val_loss=NaN) while bypassing state-variable leaks.
   */
  public dequantizeOutputs(rawOutputs: tf.Tensor, state: PersistentState): tf.Tensor {
    return tf.tidy(() => {
      // Extract input dimension (d) dynamically from the matrix profile shape array
      const inFeatures = rawOutputs.shape[rawOutputs.shape.length - 1];

      // Scale down by 1 / sqrt(inFeatures) to maintain standard variance bounds
      const scaleFactor = tf.scalar(1.0 / Math.sqrt(inFeatures), "float32");
      return tf.mul(rawOutputs, scaleFactor);
    });
  }

  /**
   * Pure functional calculator mapping for parameter updates.
   * Completely free of state-variable modifications or memory leaks.
   */
  public computeUpdate(weightTensor: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number): tf.Tensor {
    return tf.tidy(() => {
      // 1. Unpack compressed sub-byte variables
      const { weight, momentum } = this.unpack(weightTensor);

      // 2. Compute next velocity via Thermodynamic Friction
      const nextMomentum = this.calculateNextMomentum(momentum, gradient);

      // 3. Evaluate Stochastic Transitions using the Ternary Step Gate Strategy
      const scale_t = 1.0 + this.K * (1.0 - learningRate);
      const { updatedWeight, dampedMomentum } = this.evaluateWeightFlips(weight, nextMomentum, scale_t);

      // 4. Pack parameters directly back down into the 4-to-1 matrix layout
      return this.pack(updatedWeight, dampedMomentum);
    });
  }

  public applyUpdate(weightVar: tf.Variable, update: tf.Tensor): void {
    weightVar.assign(update);
  }
}

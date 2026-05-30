import * as tf from "@tensorflow/tfjs";

export interface BitNetLayerConfig {
  inFeatures: number;
  outFeatures: number;
  gradScale?: number;
  K?: number;
}

export class FlatFusedBitNetLayer {
  public readonly inFeatures: number;
  public readonly outFeatures: number;
  public readonly gradScale: number;
  public readonly K: number;

  // Strict type tracking for our unified 1-byte state memory
  private packedState: tf.Tensor2D;
  private gamma: number = 1.0;

  // Pre-allocated static Look-Up Table tensor to preserve GPU memory allocation lines
  private static readonly LOG_PROB_LUT_DATA: number[] = [1.0, 0.8, 0.64, 0.512, 0.41, 0.328, 0.262, 0.21, 0.168, 0.134, 0.107, 0.086, 0.069, 0.055, 0.044, 0.035, 0.028, 0.023, 0.018, 0.014, 0.012, 0.009, 0.007, 0.006, 0.005, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  private static readonly LUT_TENSOR: tf.Tensor1D = tf.tensor1d(FlatFusedBitNetLayer.LOG_PROB_LUT_DATA, "float32");

  constructor(config: BitNetLayerConfig) {
    this.inFeatures = config.inFeatures;
    this.outFeatures = config.outFeatures;
    this.gradScale = config.gradScale ?? 10.0;
    this.K = config.K ?? 0.5;

    // Initialize state matrix fully fused: Weights at '01' (Neutral 0), Momentum at 0
    const totalParams = this.inFeatures * this.outFeatures;
    const initialStates = new Int32Array(totalParams).fill(1); // 1 = Weight binary '01'

    this.packedState = tf.tensor1d(initialStates, "int32").reshape([this.inFeatures, this.outFeatures]) as tf.Tensor2D;
  }

  /**
   * UNIT 1: INTERNAL REGISTER EXTRACTION (Read-Only)
   * REFACTOR: Universal Arithmetic (Replaces missing bitwise ops)
   */
  private static unpackState(fused: tf.Tensor2D): { weight: tf.Tensor2D; momentum: tf.Tensor2D } {
    return tf.tidy(() => {
      // 1. Extract Weight: (fused & 0x03)
      //    Mathematically equivalent to fused % 4
      const weight = tf.mod(fused, tf.scalar(4, "int32")) as tf.Tensor2D;

      // 2. Extract Momentum: (fused >> 2)
      //    Mathematically equivalent to floor(fused / 4)
      const rawShifted = tf.floor(tf.div(fused.toFloat(), tf.scalar(4.0, "float32"))).toInt();

      //    Mask to 6-bit unsigned (rawShifted & 0x3F)
      //    Mathematically equivalent to rawShifted % 64
      const unsignedMom = tf.mod(rawShifted, tf.scalar(64, "int32"));

      // 3. Manual Sign Extension (Branchless Two's Complement Restore)
      //    If value >= 32, it's negative in 6-bit land. Subtract 64 to restore the sign.
      const isNegative = tf.greaterEqual(unsignedMom, tf.scalar(32, "int32"));
      const momentum = tf.where(isNegative, tf.sub(unsignedMom, tf.scalar(64, "int32")), unsignedMom) as tf.Tensor2D;

      return { weight, momentum };
    });
  }

  /**
   * UNIT 1: INTERNAL REGISTER COMPRESSION (Write-Back)
   * REFACTOR: Universal Arithmetic (Replaces missing bitwise ops)
   */
  private static packState(weight: tf.Tensor2D, momentum: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      // 1. Constrain boundaries
      const clampedWeight = tf.clipByValue(weight, 0, 2);
      const clampedMomentum = tf.clipByValue(momentum, -32, 31);

      // 2. Mask momentum to 6-bit Unsigned (clampedMomentum & 0x3F)
      //    For positive numbers (0-31), this does nothing.
      //    For negative numbers (-32 to -1), we must add 64 to get the unsigned binary form.
      //    Example: -1 becomes 63 (111111), -32 becomes 32 (100000).
      const isMomNeg = tf.less(clampedMomentum, tf.scalar(0, "int32"));
      const unsignedMomentum = tf.where(isMomNeg, tf.add(clampedMomentum, tf.scalar(64, "int32")), clampedMomentum);

      // 3. Shift momentum UP by 2 bits (unsignedMomentum << 2)
      //    Mathematically equivalent to unsignedMomentum * 4
      const shiftedMomentum = tf.mul(unsignedMomentum, tf.scalar(4, "int32"));

      // 4. Fuse using Addition (Replaces Bitwise OR)
      //    Since 'weight' occupies bits 0-1 and 'shiftedMomentum' occupies bits 2-7,
      //    they are mathematically disjoint. Addition is safe.
      return tf.add(clampedWeight, shiftedMomentum) as tf.Tensor2D;
    });
  }

  /**
   * UNIT 2: LAYER 1 STOCHASTIC GRADIENT RECTIFICATION
   */
  private static updateMomentum(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D {
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
      const brakeStep = tf.scalar(-3.0, "float32");

      const finalStepDeltaFloat = tf.where(activeAcceleration, tf.mul(accelStep, gradSign), tf.mul(brakeStep, momSign));

      const updatedMomUnclamped = tf.add(currentMomentum, finalStepDeltaFloat.toInt());
      return tf.clipByValue(updatedMomUnclamped, -32, 31) as tf.Tensor2D;
    });
  }

  /**
   * UNIT 3: LAYER 2 ANNEALED LOGARITHMIC WEIGHT CONVERSION
   */
  private static updateWeightsAndDamp(weight: tf.Tensor2D, momentum: tf.Tensor2D, scale_t: number): { updatedWeight: tf.Tensor2D; dampedMomentum: tf.Tensor2D } {
    return tf.tidy(() => {
      const absVelocity = tf.abs(momentum);
      const annealedIndexFloat = tf.mul(absVelocity.toFloat(), tf.scalar(scale_t, "float32"));
      const annealedIndex = tf.clipByValue(annealedIndexFloat.toInt(), 0, 32);

      const flipProbability = tf.gather(FlatFusedBitNetLayer.LUT_TENSOR, annealedIndex);
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

  /**
   * PUBLIC EXPOSED INTERFACE: CLEAN STE FORWARD PASS
   * FINAL FIX:
   * 1. Unwraps tf.tidy to allow returning the 'gradFunc' function.
   * 2. Returns 'realWeights' from inner tidy to prevent premature disposal.
   * 3. Uses 'save()' to persist weights for the backward pass.
   */
  public forward(x: tf.Tensor2D): tf.Tensor2D {
    const customOp = tf.customGrad((...args: any[]): { value: tf.Tensor; gradFunc: (dy: tf.Tensor, saved: tf.Tensor[]) => tf.Tensor | tf.Tensor[] } => {
      // 1. Extract arguments (Input Tensor and Save Function)
      const xInput = args[0] as tf.Tensor2D;
      const save = args[args.length - 1] as tf.GradSaveFunc;

      // 2. Execute Forward Math inside tf.tidy to contain intermediates
      //    We MUST return both the output and the unpacked weights to keep them alive.
      const fwdResult = tf.tidy(() => {
        const { weight } = FlatFusedBitNetLayer.unpackState(this.packedState);

        // Map (0,1,2) -> (-1.0, 0.0, +1.0)
        const realWeights = tf.sub(weight.toFloat(), tf.scalar(1.0, "float32"));

        const output = tf.matMul(xInput, realWeights);
        const scaledOutput = tf.mul(output, tf.scalar(this.gamma, "float32")) as tf.Tensor2D;

        // Return both so tidy doesn't kill them
        return { scaledOutput, realWeights };
      });

      // 3. Save the weights for the backward pass
      //    (TensorFlow.js will now manage their lifecycle until backprop is done)
      save([fwdResult.realWeights]);

      // 4. Define Gradient Function using the SAVED weights
      const gradFunc = (dy: tf.Tensor, saved: tf.Tensor[]) => {
        return tf.tidy(() => {
          const [savedWeights] = saved; // Retrieve the weights we saved
          const dy2D = dy as tf.Tensor2D;
          const savedWeights2D = savedWeights as tf.Tensor2D;

          // Straight-Through Estimator: Pass gradient back through the ternary weights
          return tf.matMul(dy2D, savedWeights2D.transpose()) as tf.Tensor2D;
        });
      };

      return { value: fwdResult.scaledOutput, gradFunc };
    });

    return customOp(x) as tf.Tensor2D;
  }

  /**
   * PUBLIC EXPOSED INTERFACE: LOGARITHMIC BACKWARD WEIGHT INTEGRATION
   */
  public applyStep(gradients: tf.Tensor2D, currentLr: number): void {
    tf.tidy(() => {
      const scale_t = 1.0 + this.K * (1.0 - currentLr);
      const { weight, momentum } = FlatFusedBitNetLayer.unpackState(this.packedState);

      const nextMomentum = FlatFusedBitNetLayer.updateMomentum(momentum, gradients, this.gradScale);
      const { updatedWeight, dampedMomentum } = FlatFusedBitNetLayer.updateWeightsAndDamp(weight, nextMomentum, scale_t);
      const nextPackedState = FlatFusedBitNetLayer.packState(updatedWeight, dampedMomentum);

      // Lazy-calculate Static Variance Normalization Scalar
      const realWeightsFloat = tf.sub(updatedWeight.toFloat(), tf.scalar(1.0, "float32"));
      const averageMagnitude = tf.mean(tf.abs(realWeightsFloat));
      const gammaData = averageMagnitude.dataSync()[0];
      this.gamma = gammaData > 0 ? gammaData : 1.0;

      // Secure Memory Block via Atomic Swap Mutation
      const oldPacked = this.packedState;
      this.packedState = tf.keep(nextPackedState);
      oldPacked.dispose();
    });
  }

  public dispose(): void {
    if (this.packedState) {
      this.packedState.dispose();
    }
  }
}

import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "../IBitNetStrategy";
import { PersistentState } from "../PersistentState";

export class FlatFusedBitNetStrategy implements IBitNetStrategy {
  private readonly gradScale: number;
  private readonly K: number;

  private static readonly LOG_PROB_LUT_DATA = [1.0, 0.8, 0.64, 0.512, 0.41, 0.328, 0.262, 0.21, 0.168, 0.134, 0.107, 0.086, 0.069, 0.055, 0.044, 0.035, 0.028, 0.023, 0.018, 0.014, 0.012, 0.009, 0.007, 0.006, 0.005, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  private static readonly LUT_TENSOR = tf.tensor1d(FlatFusedBitNetStrategy.LOG_PROB_LUT_DATA, "float32");

  constructor(config?: { gradScale?: number; K?: number }) {
    this.gradScale = config?.gradScale ?? 10.0;
    this.K = config?.K ?? 0.5;
  }

  public getPackedShape(outFeatures: number, inFeatures: number): tf.Shape {
    return [inFeatures, outFeatures];
  }

  public prepareInitialWeights(rawFloatWeights: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const beta = tf.maximum(tf.mean(tf.abs(rawFloatWeights)), tf.scalar(1e-5));
      const scaled = tf.div(rawFloatWeights, beta);
      const ternaryRaw = tf.clipByValue(tf.round(scaled), -1, 1);

      // Map (-1, 0, 1) -> (0, 1, 2)
      const unsignedWeights = tf.add(ternaryRaw, tf.scalar(1.0, "float32")).toInt();
      const initialMomentum = tf.zerosLike(unsignedWeights);

      // Pack them together and cast back to float32 to match the layer's variable type
      return this.packState(unsignedWeights, initialMomentum).toFloat();
    });
  }

  private unpackState(fusedFloat: tf.Tensor): { weight: tf.Tensor; momentum: tf.Tensor } {
    // ADJUSTMENT: Cast the storage float32 back to int32 before extracting bits
    const fused = fusedFloat.toInt();

    // 1. Extract Weight: (fused & 0x03) -> fused % 4
    const weight = tf.mod(fused, tf.scalar(4, "int32"));

    // 2. Extract Momentum: (fused >> 2) -> floor(fused / 4)
    const rawShifted = tf.floor(tf.div(fused.toFloat(), tf.scalar(4.0, "float32"))).toInt();
    const unsignedMom = tf.mod(rawShifted, tf.scalar(64, "int32"));

    // 3. Two's Complement Sign Restoration
    const isNegative = tf.greaterEqual(unsignedMom, tf.scalar(32, "int32"));
    const momentum = tf.where(isNegative, tf.sub(unsignedMom, tf.scalar(64, "int32")), unsignedMom);

    return { weight, momentum };
  }

  private packState(weight: tf.Tensor, momentum: tf.Tensor): tf.Tensor {
    const clampedWeight = tf.clipByValue(weight, 0, 2);
    const clampedMomentum = tf.clipByValue(momentum, -31, 31);

    const isMomNeg = tf.less(clampedMomentum, tf.scalar(0, "int32"));
    const unsignedMomentum = tf.where(isMomNeg, tf.add(clampedMomentum, tf.scalar(64, "int32")), clampedMomentum);

    const shiftedMomentum = tf.mul(unsignedMomentum, tf.scalar(4, "int32"));
    return tf.add(clampedWeight, shiftedMomentum);
  }

  public decodeWeights(packedTensor: tf.Tensor, state: PersistentState): tf.Tensor {
    const { weight } = this.unpackState(packedTensor);
    const realWeights = tf.sub(weight.toFloat(), tf.scalar(1.0, "float32"));

    const averageMagnitude = tf.mean(tf.abs(realWeights));
    const gamma = tf.where(tf.greater(averageMagnitude, 0), averageMagnitude, tf.scalar(1.0, "float32"));
    state.set("gamma", gamma);

    return realWeights;
  }

  public quantizeActivations(inputs: tf.Tensor, state: PersistentState): tf.Tensor {
    return inputs; // Pass-through for this specific fused strategy
  }

  public dequantizeOutputs(rawOutputs: tf.Tensor, state: PersistentState): tf.Tensor {
    const gamma = state.get("gamma");
    if (!gamma) throw new Error("Dequantization failed: Gamma is missing.");
    return tf.mul(rawOutputs, gamma);
  }

  public computeUpdate(weightTensor: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number): tf.Tensor {
    return tf.tidy(() => {
      const { weight, momentum } = this.unpackState(weightTensor);

      // 1. Stochastic Gradient Rectification
      const momFloat = momentum.toFloat();
      const momSign = tf.sign(momFloat);
      const gradSign = tf.neg(tf.sign(gradient));
      const gradMagnitude = tf.abs(gradient);

      const shouldAccelerate = tf.logicalOr(tf.equal(gradSign, momSign), tf.equal(momSign, tf.scalar(0.0, "float32")));
      const isGradActive = tf.notEqual(gradSign, tf.scalar(0.0, "float32"));
      const activeAcceleration = tf.logicalAnd(shouldAccelerate, isGradActive);

      const rawProb = tf.mul(gradMagnitude, tf.scalar(this.gradScale, "float32"));
      const randSlice = tf.randomUniform(gradient.shape, 0, 1, "float32");
      const accelStep = tf.where(tf.less(randSlice, tf.clipByValue(rawProb, 0, 1)), tf.scalar(1.0, "float32"), tf.scalar(0.0, "float32"));

      // FIX: Explicitly enforce float32 bounds inside where(), then map to int32 immediately
      const finalStepDeltaFloat = tf.where(activeAcceleration, tf.mul(accelStep, gradSign), tf.mul(tf.scalar(-3.0, "float32"), momSign));
      const nextMomentum = tf.clipByValue(tf.add(momentum, finalStepDeltaFloat.toInt()), -31, 31);

      // 2. Annealed Logarithmic Weight Conversion
      const scale_t = 1.0 + this.K * (1.0 - learningRate);
      const annealedIndex = tf.clipByValue(tf.mul(tf.abs(nextMomentum).toFloat(), tf.scalar(scale_t, "float32")).toInt(), 0, 32);
      const flipProbability = tf.gather(FlatFusedBitNetStrategy.LUT_TENSOR, annealedIndex);
      const shouldFlip = tf.less(tf.randomUniform(momentum.shape, 0, 1, "float32"), flipProbability);

      const isMomPos = tf.greater(nextMomentum, tf.scalar(0, "int32"));
      const isMomNeg = tf.less(nextMomentum, tf.scalar(0, "int32"));

      const weightIncrement = tf.logicalAnd(tf.logicalAnd(shouldFlip, isMomPos), tf.notEqual(weight, tf.scalar(2, "int32"))).toInt();
      const weightDecrement = tf.logicalAnd(tf.logicalAnd(shouldFlip, isMomNeg), tf.notEqual(weight, tf.scalar(0, "int32"))).toInt();
      const updatedWeight = tf.add(tf.sub(weight, weightDecrement), weightIncrement);

      // 3. Apply Dampening Checks safely with uniform int32 tensors
      const actualFlipOccurred = tf.logicalOr(tf.greater(weightIncrement, tf.scalar(0, "int32")), tf.greater(weightDecrement, tf.scalar(0, "int32")));
      const positiveDamp = tf.clipByValue(tf.sub(nextMomentum, tf.scalar(3, "int32")), 0, 31);
      const negativeDamp = tf.clipByValue(tf.add(nextMomentum, tf.scalar(3, "int32")), -31, 0);

      // FIX: Both arguments are explicitly int32 tensors now
      const dampedState = tf.where(isMomPos, positiveDamp, negativeDamp);
      const dampedMomentum = tf.where(actualFlipOccurred, dampedState, nextMomentum);

      // 4. Pack together and pass back to the float32 weight container
      return this.packState(updatedWeight, dampedMomentum).toFloat();
    });
  }

  public applyUpdate(weightVar: tf.Variable, update: tf.Tensor): void {
    weightVar.assign(update);
  }
}

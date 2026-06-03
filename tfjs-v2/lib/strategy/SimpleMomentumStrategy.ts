import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "./IBitNetStrategy";

export class SimpleMomentumStrategy implements IBitNetStrategy {
  private static readonly LOG_PROB_LUT_DATA = [0.0, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.044, 0.055, 0.069, 0.086, 0.107, 0.134, 0.168, 0.21, 0.262, 0.328, 0.41, 0.512, 0.64, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

  private lut64: tf.Tensor1D;
  private momentumDecay: number;

  constructor(momentumDecay = 0.9) {
    this.momentumDecay = momentumDecay;

    this.lut64 = tf.tidy(() => {
      const expanded = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        const physicalVelocity = i - 32;
        const magnitude = Math.min(32, Math.abs(physicalVelocity));
        expanded[i] = SimpleMomentumStrategy.LOG_PROB_LUT_DATA[magnitude];
      }
      return tf.tensor1d(expanded, "float32");
    });
  }

  public prepareInitialWeights(rawFloatWeights: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const quantizedTernary = tf.clipByValue(tf.round(rawFloatWeights), -1, 1);
      const zeroVelocity = tf.zerosLike(quantizedTernary).toInt();
      return this.pack(zeroVelocity, tf.cast(quantizedTernary, "int32"));
    });
  }

  public unpack(packedTensor: tf.Tensor): { ternaryWeights: tf.Tensor; velocity: tf.Tensor } {
    return tf.tidy(() => {
      const velocity = tf.cast(tf.floor(tf.div(tf.cast(packedTensor, "float32"), 4.0)), "int32");

      const positivePacked = tf.add(tf.mod(packedTensor, 4), 4);
      const lowerTwoBits = tf.mod(positivePacked, 4);

      const zeroTensor = tf.zerosLike(lowerTwoBits);
      const oneTensor = tf.onesLike(lowerTwoBits);
      const negativeOneTensor = tf.mul(-1.0, oneTensor);

      const ternaryWeights = tf.where(tf.equal(lowerTwoBits, 0), zeroTensor, tf.where(tf.equal(lowerTwoBits, 1), oneTensor, negativeOneTensor));

      return { ternaryWeights, velocity };
    });
  }

  /**
   * High-efficiency mathematical mutation pass.
   */
  public mutate(currentVelocity: tf.Tensor, currentTernaryWeights: tf.Tensor, gradient: tf.Tensor, learningRate: number) {
    return tf.tidy(() => {
      const decayedVelocity = tf.mul(tf.cast(currentVelocity, "float32"), this.momentumDecay);
      const continuousDeltaV = tf.mul(-learningRate, tf.cast(gradient, "float32"));
      const totalContinuousUpdate = tf.add(decayedVelocity, continuousDeltaV);

      const floorV = tf.floor(totalContinuousUpdate);
      const remainderV = tf.sub(totalContinuousUpdate, floorV);
      const randomMatrix = tf.randomUniform(totalContinuousUpdate.shape, 0, 1);
      const stochasticStep = tf.cast(tf.less(randomMatrix, remainderV), "float32");

      const clippedVelocity = tf.clipByValue(tf.add(floorV, stochasticStep), -31, 31);
      const newDiscreteVelocity = tf.cast(clippedVelocity, "int32");

      const lutIndices = tf.add(newDiscreteVelocity, 32);
      const flatIndices = tf.cast(tf.reshape(lutIndices, [-1]), "int32");
      const flatProbabilities = tf.gather(this.lut64, flatIndices);
      const probabilityMatrix = tf.reshape(flatProbabilities, newDiscreteVelocity.shape);

      const diceRoll = tf.randomUniform(newDiscreteVelocity.shape, 0, 1);
      const shouldMove = tf.less(diceRoll, probabilityMatrix);

      const onesTensor = tf.onesLike(newDiscreteVelocity).toInt();
      const negativeOnesTensor = tf.mul(tf.scalar(-1, "int32"), onesTensor);

      const movementDirection = tf.where(tf.less(newDiscreteVelocity, 0), onesTensor, tf.where(tf.greater(newDiscreteVelocity, 0), negativeOnesTensor, tf.zerosLike(newDiscreteVelocity).toInt()));

      const appliedStep = tf.mul(tf.cast(shouldMove, "int32"), movementDirection);
      const proposedWeights = tf.add(tf.cast(currentTernaryWeights, "int32"), appliedStep);
      const finalTernaryWeights = tf.clipByValue(proposedWeights, -1, 1);

      const hitHardBoundary = tf.notEqual(proposedWeights, finalTernaryWeights);

      const dampedVelocity = tf.cast(tf.floor(tf.div(tf.cast(newDiscreteVelocity, "float32"), 2.0)), "int32");
      const finalVelocity = tf.where(hitHardBoundary, dampedVelocity, newDiscreteVelocity);

      return {
        finalVelocity,
        finalTernaryWeights: tf.cast(finalTernaryWeights, "int32"),
      };
    });
  }

  /**
   * High-efficiency packing via linear arithmetic instead of conditional Select graphs.
   */
  public pack(velocity: tf.Tensor, ternaryWeights: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const safeVelocity = tf.clipByValue(velocity, -31, 31);
      const shiftedVelocity = tf.mul(safeVelocity, 4);

      // Highly efficient pure arithmetic mapping to isolate lower bits cleanly
      const isNegative = tf.cast(tf.less(ternaryWeights, 0), "int32");
      const isPositive = tf.cast(tf.greater(ternaryWeights, 0), "int32");
      const positiveWeightMask = tf.add(tf.mul(isNegative, 2), isPositive);

      return tf.add(shiftedVelocity, positiveWeightMask);
    });
  }

  static get className(): string {
    return "SimpleMomentumStrategy";
  }
}

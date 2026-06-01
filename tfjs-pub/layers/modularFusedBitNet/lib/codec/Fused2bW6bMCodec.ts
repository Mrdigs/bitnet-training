import { ParameterStorageCodec } from "../interfaces";
import * as tf from "@tensorflow/tfjs";

/**
 * Fused2bW6bMCodec
 *
 * Default codec for packing and unpacking parameter state. Each element
 * encodes a ternary token in bits [0:1] (values 0/1/2 for -1/0/+1) and
 * a signed 6-bit momentum in bits [2:7] (range -31..+31). The result is
 * represented as `int32` for tensor safety.
 *
 * Unlike `DynamicSymmetricalCodec`, this codec does not accept initial
 * weights; `getInitialState` simply fills the matrix with token value 1
 * (the neutral/zero token) and zero momentum.
 *
 * TODO: Maybe it *should* accept initial weights.
 */
export class Fused2bW6bMCodec implements ParameterStorageCodec {
  private initialWeights: tf.Tensor2D;

  constructor(initialWeights: tf.Tensor2D) {
    // Require initialWeights to be provided and keep a clone to manage lifecycle
    this.initialWeights = tf.keep(initialWeights.clone());
  }

  public unpack(fused: tf.Tensor2D): { weight: tf.Tensor2D; momentum: tf.Tensor2D } {
    return tf.tidy(() => {
      const weight = tf.mod(fused, tf.scalar(4, "int32")) as tf.Tensor2D;
      const rawShifted = tf.floor(tf.div(fused.toFloat(), tf.scalar(4.0, "float32"))).toInt();
      const unsignedMom = tf.mod(rawShifted, tf.scalar(64, "int32"));
      const isNegative = tf.greaterEqual(unsignedMom, tf.scalar(32, "int32"));
      const momentum = tf.where(isNegative, tf.sub(unsignedMom, tf.scalar(64, "int32")), unsignedMom) as tf.Tensor2D;
      return { weight, momentum };
    });
  }

  public pack(weight: tf.Tensor2D, momentum: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      // Ensure integer arithmetic so packed values fit into 8 bits (0..255)
      const clampedWeight = tf.clipByValue(weight, 0, 2);
      const clampedWeightInt = clampedWeight.toInt();

      const clampedMomentum = tf.clipByValue(momentum, -31, 31);
      const clampedMomentumInt = clampedMomentum.toInt();

      const isMomNeg = tf.less(clampedMomentumInt, tf.scalar(0, "int32"));
      const unsignedMomentum = tf.where(isMomNeg, tf.add(clampedMomentumInt, tf.scalar(64, "int32")), clampedMomentumInt) as tf.Tensor2D;
      const shiftedMomentum = tf.mul(unsignedMomentum, tf.scalar(4, "int32"));

      // fused value in range 0..254 fits within a byte
      const fused = tf.add(clampedWeightInt, shiftedMomentum) as tf.Tensor2D;
      return fused;
    });
  }

  public getInitialState(inFeatures: number, outFeatures: number): tf.Tensor2D {
    return tf.tidy(() => {
      // Validate shape
      const shape = this.initialWeights.shape;
      if (shape[0] !== inFeatures || shape[1] !== outFeatures) {
        throw new Error(`Fused2bW6bMCodec: initialWeights shape ${shape} does not match requested [${inFeatures}, ${outFeatures}]`);
      }

      // Map continuous weights to ternary tokens as integer tensor: negative->0, zero->1, positive->2
      const negMask = tf.less(this.initialWeights, tf.scalar(0, "float32"));
      const posMask = tf.greater(this.initialWeights, tf.scalar(0, "float32"));
      const zeros = tf.zeros(this.initialWeights.shape, "int32") as tf.Tensor2D;
      const twos = tf.fill(this.initialWeights.shape, 2, "int32") as tf.Tensor2D;
      const ones = tf.fill(this.initialWeights.shape, 1, "int32") as tf.Tensor2D;
      const tokens = tf.where(negMask, zeros, tf.where(posMask, twos, ones)) as tf.Tensor2D;
      // Pack tokens with zero momentum
      const zeroMomentum = tf.zeros(this.initialWeights.shape, "int32") as tf.Tensor2D;
      return this.pack(tokens as unknown as tf.Tensor2D, zeroMomentum);
    });
  }

  public dispose(): void {
    if (this.initialWeights) {
      this.initialWeights.dispose();
    }
  }
}

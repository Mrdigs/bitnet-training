import { ParameterStorageCodec } from "../interfaces";
import * as tf from "@tensorflow/tfjs";

/**
 * Fused2bW6bMCodec
 *
 * Default codec for packing and unpacking parameter state. Each element
 * encodes a ternary token in bits [0:1] (values 0/1/2 for -1/0/+1) and
 * a signed 6-bit momentum in bits [2:7] (range -32..+31). The result is
 * represented as `int32` for tensor safety.
 *
 * Unlike `DynamicSymmetricalCodec`, this codec does not accept initial
 * weights; `getInitialState` simply fills the matrix with token value 1
 * (the neutral/zero token) and zero momentum.
 *
 * TODO: Maybe it *should* accept initial weights.
 */
export class Fused2bW6bMCodec implements ParameterStorageCodec {
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
      const clampedWeight = tf.clipByValue(weight, 0, 2);
      const clampedMomentum = tf.clipByValue(momentum, -32, 31);
      const isMomNeg = tf.less(clampedMomentum, tf.scalar(0, "int32"));
      const unsignedMomentum = tf.where(isMomNeg, tf.add(clampedMomentum, tf.scalar(64, "int32")), clampedMomentum);
      const shiftedMomentum = tf.mul(unsignedMomentum, tf.scalar(4, "int32"));
      return tf.add(clampedWeight, shiftedMomentum) as tf.Tensor2D;
    });
  }

  public getInitialState(inFeatures: number, outFeatures: number): tf.Tensor2D {
    const totalParams = inFeatures * outFeatures;
    const initialStates = new Int32Array(totalParams).fill(1); // Default to token '01' (Mathematical Zero)
    return tf.tensor1d(initialStates, "int32").reshape([inFeatures, outFeatures]) as tf.Tensor2D;
  }
}

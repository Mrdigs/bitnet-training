import * as tf from "@tensorflow/tfjs";
import { ParameterStorageCodec } from "../interfaces";

export class DynamicSymmetricalCodec implements ParameterStorageCodec {
  private initialWeights: tf.Tensor2D | null;

  constructor(initialWeights: tf.Tensor2D | null = null) {
    this.initialWeights = initialWeights;
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
      const clampedWeight = tf.clipByValue(weight, 0, 2);
      const clampedMomentum = tf.clipByValue(momentum, -31, 31);
      const isMomNeg = tf.less(clampedMomentum, tf.scalar(0, "int32"));
      const unsignedMomentum = tf.where(isMomNeg, tf.add(clampedMomentum, tf.scalar(64, "int32")), clampedMomentum);
      const shiftedMomentum = tf.mul(unsignedMomentum, tf.scalar(4, "int32"));
      return tf.add(clampedWeight, shiftedMomentum) as tf.Tensor2D;
    });
  }

  public getInitialState(inFeatures: number, outFeatures: number): tf.Tensor2D {
    return tf.tidy(() => {
      if (!this.initialWeights) {
        return tf.ones([inFeatures, outFeatures], "int32") as tf.Tensor2D;
      }

      const wFloats = this.initialWeights;

      // FIX: Auto-calculate the variance threshold dynamically from the matrix physics
      const stdDev = tf.sqrt(tf.mean(tf.square(tf.sub(wFloats, tf.mean(wFloats)))));
      const threshold = tf.mul(stdDev, tf.scalar(0.65, "float32"));

      // Balanced ternary distribution: ~33% positive, ~33% negative, ~33% neutral
      const isNegativeZone = tf.less(wFloats, tf.neg(threshold));
      const isPositiveZone = tf.greater(wFloats, threshold);

      const tokens = tf.where(isNegativeZone, tf.scalar(0, "int32"), tf.where(isPositiveZone, tf.scalar(2, "int32"), tf.scalar(1, "int32")));

      const clampedMomentum = tf.zeros([inFeatures, outFeatures], "int32");
      const shiftedMomentum = tf.mul(clampedMomentum, tf.scalar(4, "int32"));

      return tf.add(tokens, shiftedMomentum) as tf.Tensor2D;
    });
  }
}

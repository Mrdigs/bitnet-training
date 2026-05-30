import * as tf from "@tensorflow/tfjs";
import { ParameterStorageCodec } from "../interfaces";

/**
 * @deprecated Use DynamicSymmetricalCodec instead for more robust initialization
 */
export class SymmetricalInitializationCodec implements ParameterStorageCodec {
  private initialAdamWeights: tf.Tensor2D | null;

  /**
   * @param initialAdamWeights The raw continuous random normal weights matrix used by Adam.
   */
  constructor(initialAdamWeights: tf.Tensor2D | null = null) {
    this.initialAdamWeights = initialAdamWeights;
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
      const clampedMomentum = tf.clipByValue(momentum, -32, 31);
      const isMomNeg = tf.less(clampedMomentum, tf.scalar(0, "int32"));
      const unsignedMomentum = tf.where(isMomNeg, tf.add(clampedMomentum, tf.scalar(64, "int32")), clampedMomentum);
      const shiftedMomentum = tf.mul(unsignedMomentum, tf.scalar(4, "int32"));
      return tf.add(clampedWeight, shiftedMomentum) as tf.Tensor2D;
    });
  }

  public getInitialState(inFeatures: number, outFeatures: number): tf.Tensor2D {
    return tf.tidy(() => {
      // Fallback to absolute zero tokens if no source matrix is provided
      if (!this.initialAdamWeights) {
        const totalParams = inFeatures * outFeatures;
        const initialStates = new Int32Array(totalParams).fill(1);
        return tf.tensor1d(initialStates, "int32").reshape([inFeatures, outFeatures]) as tf.Tensor2D;
      }

      // DETERMINISTIC QUANTIZATION GATE: Map Adam continuous floats straight to 0, 1, 2 tokens
      // val < -0.5 -> 0 (-1), val > 0.5 -> 2 (+1), else 1 (0)
      const wFloats = this.initialAdamWeights;
      const isNegativeZone = tf.less(wFloats, tf.scalar(-0.5, "float32"));
      const isPositiveZone = tf.greater(wFloats, tf.scalar(0.5, "float32"));

      const tokens = tf.where(isNegativeZone, tf.scalar(0, "int32"), tf.where(isPositiveZone, tf.scalar(2, "int32"), tf.scalar(1, "int32")));

      // Momentum bits are kept completely at neutral 0. Pack elements cleanly.
      const clampedMomentum = tf.zeros([inFeatures, outFeatures], "int32");
      const shiftedMomentum = tf.mul(clampedMomentum, tf.scalar(4, "int32"));

      return tf.add(tokens, shiftedMomentum) as tf.Tensor2D;
    });
  }
}

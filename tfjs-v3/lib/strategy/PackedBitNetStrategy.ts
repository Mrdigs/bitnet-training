import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "../IBitNetStrategy";

export abstract class PackedBitNetStrategy {
  /**
   * Calculates the compressed row dimension required for storage.
   * Reduces the vertical input dimension footprint by 75%.
   */
  public getPackedShape(outFeatures: number, inFeatures: number): tf.Shape {
    return [Math.ceil(inFeatures / 4), outFeatures];
  }

  /**
   * Packs four vertical slices of uncompressed float32 weights/momenta
   * into a single packed float32 tensor layer.
   *
   * Expects weight slices in range [0, 2] and momentum slices in range [-31, 31].
   */
  public pack(weights: tf.Tensor2D[], momenta: tf.Tensor2D[]): tf.Tensor2D {
    return tf.tidy(() => {
      const bytePackets: tf.Tensor[] = [];

      for (let i = 0; i < 4; i++) {
        const w = weights[i];
        const m = momenta[i];

        // Map signed momentum [-31, 31] to unsigned byte values [0, 63]
        const isMomNeg = tf.less(m, tf.scalar(0.0, "float32"));
        const unsignedMom = tf.where(isMomNeg, tf.add(m, tf.scalar(64.0, "float32")), m);
        const shiftedMom = tf.mul(unsignedMom, tf.scalar(4.0, "float32"));

        // Combine clamped weight and shifted momentum into a single 8-bit byte configuration
        const bytePacket = tf.add(tf.clipByValue(w, 0.0, 2.0), shiftedMom);
        bytePackets.push(bytePacket);
      }

      // Universal Arithmetic Packing: 4 bytes -> single float32 register
      return tf.add(bytePackets[0], tf.add(tf.mul(bytePackets[1], tf.scalar(256.0, "float32")), tf.add(tf.mul(bytePackets[2], tf.scalar(65536.0, "float32")), tf.mul(bytePackets[3], tf.scalar(16777216.0, "float32"))))) as tf.Tensor2D;
    });
  }

  /**
   * Unpacks a single compressed float32 matrix back into 4 separate vertical Byte packets.
   */
  public unpackBytes(fused: tf.Tensor2D): tf.Tensor2D[] {
    return tf.tidy(() => {
      const b0 = tf.mod(fused, tf.scalar(256.0, "float32"));
      const b1 = tf.mod(tf.floor(tf.div(fused, tf.scalar(256.0, "float32"))), tf.scalar(256.0, "float32"));
      const b2 = tf.mod(tf.floor(tf.div(fused, tf.scalar(65536.0, "float32"))), tf.scalar(256.0, "float32"));
      const b3 = tf.floor(tf.div(fused, tf.scalar(16777216.0, "float32")));

      return [b0 as tf.Tensor2D, b1 as tf.Tensor2D, b2 as tf.Tensor2D, b3 as tf.Tensor2D];
    });
  }

  /**
   * Decodes a specific extracted byte packet into its separate Weight and Momentum tensor properties.
   */
  public decodeByteComponents(packet: tf.Tensor2D): { weight: tf.Tensor2D; momentum: tf.Tensor2D } {
    return tf.tidy(() => {
      const weight = tf.mod(packet, tf.scalar(4.0, "float32")) as tf.Tensor2D;

      const rawShifted = tf.floor(tf.div(packet, tf.scalar(4.0, "float32")));
      const unsignedMom = tf.mod(rawShifted, tf.scalar(64.0, "float32"));

      const isNegative = tf.greaterEqual(unsignedMom, tf.scalar(32.0, "float32"));
      const momentum = tf.where(isNegative, tf.sub(unsignedMom, tf.scalar(64.0, "float32")), unsignedMom) as tf.Tensor2D;

      return { weight, momentum };
    });
  }

  /**
   * Segments an upstream full-size gradient matrix [inFeatures, outFeatures]
   * vertically into 4 equal chunks matching the packed rows.
   */
  public sliceGradients(fullGrads: tf.Tensor2D, packedRows: number, outFeatures: number): tf.Tensor2D[] {
    return tf.tidy(() => {
      const gradSlices: tf.Tensor2D[] = [];
      const fullGradsRows = fullGrads.shape[0];

      for (let i = 0; i < 4; i++) {
        const startRow = i * packedRows;
        if (startRow < fullGradsRows) {
          const size = Math.min(packedRows, fullGradsRows - startRow);
          let gSlice = tf.slice(fullGrads, [startRow, 0], [size, outFeatures]);

          // Pad terminal gradient slices with zeros if they don't cleanly fit execution bounds
          if (size < packedRows) {
            gSlice = tf.concat([gSlice, tf.zeros([packedRows - size, outFeatures], "float32")], 0);
          }
          gradSlices.push(gSlice);
        } else {
          gradSlices.push(tf.zeros([packedRows, outFeatures], "float32"));
        }
      }
      return gradSlices;
    });
  }
}

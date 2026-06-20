import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "../IBitNetStrategy";
import { PersistentState } from "../PersistentState";

export abstract class FourTo1BitPackingStrategy implements IBitNetStrategy {
  // 1. PRE-ALLOCATED MODULO DIVISORS (Replacing Bitwise AND Masks to leverage native shape broadcasting)
  private static readonly DIVISOR_BYTE = tf.scalar(256, "int32"); // Emulates masking lowest 8 bits
  private static readonly DIVISOR_WEIGHT = tf.scalar(4, "int32"); // Emulates masking lowest 2 bits
  private static readonly DIVISOR_MOMENTUM = tf.scalar(64, "int32"); // Emulates masking 6 bits for state

  // 2. PRE-ALLOCATED MULTIPLIERS (Emulating Left-Shifts via Vectorized Multiplication)
  private static readonly SHIFT_L_WEIGHT_TO_MOM = tf.scalar(4, "int32"); // << 2
  private static readonly SHIFT_L_B1 = tf.scalar(256, "int32"); // << 8
  private static readonly SHIFT_L_B2 = tf.scalar(65536, "int32"); // << 16
  private static readonly SHIFT_L_B3 = tf.scalar(16777216, "int32"); // << 24

  // 3. PRE-ALLOCATED DENOMINATORS (Emulating Right-Shifts via Vectorized Floor Division)
  private static readonly SHIFT_R_WEIGHT_TO_MOM = tf.scalar(4, "int32"); // >> 2
  private static readonly SHIFT_R_B1 = tf.scalar(256, "int32"); // >> 8
  private static readonly SHIFT_R_B2 = tf.scalar(65536, "int32"); // >> 16
  private static readonly SHIFT_R_B3 = tf.scalar(16777216, "int32"); // >> 24

  // 4. ARITHMETIC SIGN BOUNDARY CONSTANTS
  private static readonly OFFSET_MOMENTUM = tf.scalar(32, "int32");

  /**
   * Utility to forcefully swap the JavaScript wrapper's type descriptor metadata.
   * Keeps the underlying physical C++ memory buffers perfectly untouched.
   */
  private spoofType(tensor: tf.Tensor, targetType: "float32" | "int32"): tf.Tensor {
    Object.defineProperty(tensor, "dtype", {
      value: targetType,
      writable: true,
      configurable: true,
      enumerable: true,
    });
    return tensor;
  }

  public getPackedShape(outFeatures: number, inFeatures: number): tf.Shape {
    if (inFeatures % 4 !== 0) {
      throw new Error(`Input features (${inFeatures}) must be perfectly divisible by 4 for packing.`);
    }
    return [inFeatures / 4, outFeatures];
  }

  /**
   * Packs weight arrays (0, 1, 2) and signed momentum arrays (-32 to 31) into
   * an int32 layout using broadcasting math, exiting disguised as a float32.
   */
  public pack(weight: tf.Tensor, momentum: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const [inFeatures, outFeatures] = weight.shape;

      const uWeight = tf.clipByValue(weight.toInt(), 0, 2);
      const sMomentum = tf.clipByValue(momentum.toInt(), -31, 31);
      const uMomentum = tf.add(sMomentum, FourTo1BitPackingStrategy.OFFSET_MOMENTUM);

      // Emulate: (uMomentum << 2) | uWeight using pre-allocated multiplication
      const shiftedMom = tf.mul(uMomentum, FourTo1BitPackingStrategy.SHIFT_L_WEIGHT_TO_MOM);
      const fusedByte = tf.mod(tf.add(uWeight, shiftedMom), FourTo1BitPackingStrategy.DIVISOR_BYTE);

      const splitRows = fusedByte.reshape([inFeatures / 4, 4, outFeatures]);

      const b0 = splitRows.gather(tf.scalar(0, "int32"), 1).reshape([inFeatures / 4, outFeatures]);
      const b1 = splitRows.gather(tf.scalar(1, "int32"), 1).reshape([inFeatures / 4, outFeatures]);
      const b2 = splitRows.gather(tf.scalar(2, "int32"), 1).reshape([inFeatures / 4, outFeatures]);
      const b3 = splitRows.gather(tf.scalar(3, "int32"), 1).reshape([inFeatures / 4, outFeatures]);

      // Emulate byte positional left-shifts via pre-allocated multipliers
      const p0 = b0;
      const p1 = tf.mul(b1, FourTo1BitPackingStrategy.SHIFT_L_B1);
      const p2 = tf.mul(b2, FourTo1BitPackingStrategy.SHIFT_L_B2);
      const p3 = tf.mul(b3, FourTo1BitPackingStrategy.SHIFT_L_B3);

      const packedIntegerTensor = tf.add(tf.add(tf.add(p0, p1), p2), p3);

      // EXIT GATE: Disguise as float32 to bypass layer variable type validation
      return this.spoofType(packedIntegerTensor, "float32");
    });
  }

  /**
   * Strips the float32 disguise, emulates right shifts using floor division,
   * and separates data cleanly via native broadcasting modulo steps.
   */
  public unpack(packedTensor: tf.Tensor): { weight: tf.Tensor; residual: tf.Tensor } {
    return tf.tidy(() => {
      const [packedIn, outFeatures] = packedTensor.shape;

      // ENTRY GATE: Restore the true int32 type interpretation right upon arrival
      const fused = this.spoofType(packedTensor.clone(), "int32");

      // Extract the 4 separate byte streams using broadcasting math
      const b0 = tf.mod(fused, FourTo1BitPackingStrategy.DIVISOR_BYTE);

      const s1 = tf.floorDiv(fused, FourTo1BitPackingStrategy.SHIFT_R_B1);
      const b1 = tf.mod(s1, FourTo1BitPackingStrategy.DIVISOR_BYTE);

      const s2 = tf.floorDiv(fused, FourTo1BitPackingStrategy.SHIFT_R_B2);
      const b2 = tf.mod(s2, FourTo1BitPackingStrategy.DIVISOR_BYTE);

      const s3 = tf.floorDiv(fused, FourTo1BitPackingStrategy.SHIFT_R_B3);
      const b3 = tf.mod(s3, FourTo1BitPackingStrategy.DIVISOR_BYTE);

      // Weave the channels atomically into a single layout allocation block
      const denseBytes = tf.stack([b0, b1, b2, b3], 1).reshape([packedIn * 4, outFeatures]);

      // Isolate weight bits and state bits using native broadcasting remainder checks
      const weight = tf.mod(denseBytes, FourTo1BitPackingStrategy.DIVISOR_WEIGHT);

      const sMom = tf.floorDiv(denseBytes, FourTo1BitPackingStrategy.SHIFT_R_WEIGHT_TO_MOM);
      const uMomentum = tf.mod(sMom, FourTo1BitPackingStrategy.DIVISOR_MOMENTUM);

      // Reverse offset arithmetic to restore standard signed parameters [-32, 31]
      const residual = tf.sub(uMomentum, FourTo1BitPackingStrategy.OFFSET_MOMENTUM);

      return { weight, residual };
    });
  }

  public abstract prepareInitialWeights(rawFloatWeights: tf.Tensor, state: PersistentState): tf.Tensor;

  public abstract decodeWeights(packedTensor: tf.Tensor, state: PersistentState): tf.Tensor;

  public abstract quantizeActivations(inputs: tf.Tensor, state: PersistentState): tf.Tensor;

  public abstract dequantizeOutputs(rawOutputs: tf.Tensor, state: PersistentState): tf.Tensor;

  public abstract computeUpdate(weightTensor: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number, currentStep: number): tf.Tensor;

  public abstract applyUpdate(weightVar: tf.Variable, update: tf.Tensor): void;
}

import * as tf from "@tensorflow/tfjs-node";
import { TwoBitSixBitAbstractStrategy } from "./TwoBitSixBitAbstractStrategy";

export class SgdMomentumStrategy extends TwoBitSixBitAbstractStrategy {
  private alpha: number;

  constructor(alpha: number = 0.9) {
    super();
    this.alpha = alpha;
  }

  /**
   * Initializes the 6-bit velocity tracking matrix to zero.
   * On our signed 6-bit grid (-32 to 31), arithmetic 0 maps straight to bits 000000.
   */
  protected override getInitialSixBitState(units: number, inFeatures: number): tf.Tensor {
    return tf.zeros([units, inFeatures], "int32");
  }

  /**
   * Executes element-wise Quantized SGD with Momentum.
   * Completely isolated to raw integer arithmetic blocks.
   */
  protected override optimizeSixBitElement(
    currentWeight: number, // Guaranteed to arrive as -1, 0, or 1
    currentSixBitState: number, // Guaranteed to arrive as an integer from 0 to 63
    gradient: number, // Continuous gradient value calculated from the tape pass
    lr: number,
  ): { newWeight: number; newSixBitState: number } {
    // 1. Map unsigned 6-bit state snapshot to signed JavaScript integers (-32 to 31)
    let velocity = currentSixBitState;
    if (velocity > 31) {
      velocity -= 64; // e.g., 63 maps cleanly back to -1
    }

    // 2. Standard Momentum Math: V_new = (alpha * V_old) - (lr * Gradient)
    // We introduce a scaling scalar multiplier (e.g., 10) to adjust the
    // sensitivity of continuous float updates against our discrete accumulator grid.
    const scaledGradientForce = gradient * lr * 10;
    const continuousVelocity = this.alpha * velocity - scaledGradientForce;

    // Smooth fractional tracking back onto nearest discrete integer coordinates
    let newVelocity = Math.round(continuousVelocity);

    // 3. Structural Accumulator Threshold Logic
    let newWeight = currentWeight;
    const threshold = 16; // Half-width boundary limit of our directional register space

    if (newVelocity >= threshold) {
      newWeight += 1;
      newVelocity = 0; // Completely flush momentum upon crossing structural boundaries
    } else if (newVelocity <= -threshold) {
      newWeight -= 1;
      newVelocity = 0;
    }

    // 4. Absolute Register Constraints Guard Clamping
    newWeight = Math.max(-1, Math.min(1, newWeight));
    newVelocity = Math.max(-32, Math.min(31, newVelocity));

    // 5. Pack signed values safely back into unsigned two's complement 6-bit block layouts
    let unsignedSixBitState = newVelocity;
    if (unsignedSixBitState < 0) {
      unsignedSixBitState += 64; // e.g., -1 maps out as 63
    }

    return {
      newWeight,
      newSixBitState: unsignedSixBitState,
    };
  }
}

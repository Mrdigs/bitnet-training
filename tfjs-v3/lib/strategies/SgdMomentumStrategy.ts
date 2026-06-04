import * as tf from "@tensorflow/tfjs-node";
import { TwoBitSixBitAbstractStrategy } from "./TwoBitSixBitAbstractStrategy";

export class SgdMomentumStrategy extends TwoBitSixBitAbstractStrategy {
  private alpha: number;
  // 3 fractional bits means our fixed-point scaling factor is 2^3 = 8
  private readonly fixedPointScale = 8;

  constructor(alpha: number = 0.9) {
    super();
    this.alpha = alpha;
  }

  protected override getInitialSixBitState(units: number, inFeatures: number): tf.Tensor {
    return tf.zeros([units, inFeatures], "int32");
  }

  protected override optimizeSixBitElement(currentWeight: number, currentSixBitState: number, gradient: number, lr: number): { newWeight: number; newSixBitState: number } {
    // 1. Convert the unsigned 6-bit state snapshot to signed layout space (-32 to 31)
    let packedVelocity = currentSixBitState;
    if (packedVelocity > 31) {
      packedVelocity -= 64;
    }

    // 2. Decode from fixed-point to get the true continuous floating-point velocity
    // e.g., if packedVelocity is 4, true velocity is 4 / 8 = 0.5
    const trueVelocity = packedVelocity / this.fixedPointScale;

    // 3. Compute Standard Momentum Update: V_new = (alpha * V_old) - (lr * Gradient)
    // No arbitrary multipliers needed anymore! The fixed-point scale naturally handles the precision.
    const continuousVelocity = this.alpha * trueVelocity - lr * gradient;

    // 4. Encode back into fixed-point by multiplying by 8 and using Math.floor to get the bit layout
    // This guarantees that fractional updates like 0.2 accumulate perfectly across iterations
    let newPackedVelocity = Math.round(continuousVelocity * this.fixedPointScale);

    // 5. Weight Flip Threshold Logic
    // We check against the actual accumulated velocity value.
    // Since our whole integer bounds span -4 to +3, a flip threshold of 2.0 or 3.0 works beautifully.
    let newWeight = currentWeight;
    const flipThreshold = 2.0;

    if (continuousVelocity >= flipThreshold) {
      newWeight += 1;
      newPackedVelocity = 0; // Flush velocity accumulator
    } else if (continuousVelocity <= -flipThreshold) {
      newWeight -= 1;
      newPackedVelocity = 0;
    }

    // 6. Strict Hardware Constraints Guard Clamping
    newWeight = Math.max(-1, Math.min(1, newWeight));
    newPackedVelocity = Math.max(-32, Math.min(31, newPackedVelocity));

    // 7. Re-encode signed two's complement integer back to an unsigned 6-bit layout byte
    let unsignedSixBitState = newPackedVelocity;
    if (unsignedSixBitState < 0) {
      unsignedSixBitState += 64;
    }

    return {
      newWeight,
      newSixBitState: unsignedSixBitState,
    };
  }
}

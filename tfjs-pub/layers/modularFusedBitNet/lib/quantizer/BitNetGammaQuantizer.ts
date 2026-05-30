import { ForwardProjectionQuantizer } from "../interfaces";
import * as tf from "@tensorflow/tfjs";

// --- QUANTIZER MODULE: Standard BitNet Variance-Normalization (Gamma) Mapping ---
export class BitNetGammaQuantizer implements ForwardProjectionQuantizer {
  public transformWeights(weightTokens: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      // Strict 2-bit ternary boundary enforcement (maps 3 down to 2)
      const safelyClamped = tf.clipByValue(weightTokens, 0, 2);
      return tf.sub(safelyClamped.toFloat(), tf.scalar(1.0, "float32"));
    });
  }

  public calculateScale(weightTokens: tf.Tensor2D): number {
    return tf.tidy(() => {
      const realWeights = this.transformWeights(weightTokens);
      const averageMagnitude = tf.mean(tf.abs(realWeights));
      const gammaData = averageMagnitude.dataSync()[0];
      return gammaData > 0 ? gammaData : 1.0;
    });
  }
}

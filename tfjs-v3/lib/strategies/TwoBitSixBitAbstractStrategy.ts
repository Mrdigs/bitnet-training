import * as tf from "@tensorflow/tfjs-node";
import { AbstractBitNetStrategy } from "./AbstractBitNetStrategy";

export abstract class TwoBitSixBitAbstractStrategy extends AbstractBitNetStrategy {
  constructor() {
    super(8, 4);
  }

  public override getPackedShape(units: number, inFeatures: number): [number, number] {
    if (inFeatures % 4 !== 0) {
      throw new Error(`InFeatures (${inFeatures}) must be a multiple of elementsPerBlock (4) for this configuration layout.`);
    }
    return [units, inFeatures / 4];
  }

  public override prepareInitialWeights(rawFloatWeights: tf.Tensor2D): tf.Tensor {
    return tf.tidy(() => {
      const clamped = tf.clipByValue(tf.round(rawFloatWeights), -1.0, 1.0);

      // 1. Map ternary weights to strictly POSITIVE unsigned 2-bit tokens
      // -1.0 -> 2,  0.0 -> 0,  1.0 -> 1
      const conditionTensor = tf.equal(clamped, -1.0);
      const twosTensor = tf.fill(clamped.shape, 2, "int32");
      const tokens = tf.where(conditionTensor, twosTensor, tf.cast(clamped, "int32"));

      // 2. Shift tokens up into bits 6-7. (tokens * 64).
      // Because tokens are strictly positive (0, 1, 2), shiftedWeights is strictly positive (0, 64, 128).
      const shiftedWeights = tf.mul(tokens, 64);

      // 3. Extract and safely clamp strategy initial 6-bit states
      const [units, inFeatures] = rawFloatWeights.shape;
      const initialStates = this.getInitialSixBitState(units, inFeatures);
      const safeInitialStates = tf.mod(tf.cast(initialStates, "int32"), 64);

      // Combined elements are guaranteed to be positive integers between 0 and 191
      const combinedElements = tf.add(shiftedWeights, safeInitialStates);

      // 4. Reshape elements into 4-way blocks [Units, InFeatures / 4, 4]
      const packedInFeatures = inFeatures / 4;
      const reshaped = tf.reshape(combinedElements, [units, packedInFeatures, 4]);
      const [w0, w1, w2, w3] = tf.split(reshaped, 4, 2);

      // 5. Pack mathematically. Squeeze the split blocks but maintain 2D columns
      const packed = tf.add(tf.add(tf.reshape(w0, [units, packedInFeatures]), tf.mul(tf.reshape(w1, [units, packedInFeatures]), 256)), tf.add(tf.mul(tf.reshape(w2, [units, packedInFeatures]), 65536), tf.mul(tf.reshape(w3, [units, packedInFeatures]), 16777216)));

      // Explicitly return a container matching your exact target packing shape layout [Units, InFeatures / 4]
      return tf.cast(packed, "float32");
    });
  }

  protected override decodeElementToToken(elementTensor: tf.Tensor): tf.Tensor {
    // Upper 2 bits track the ternary weight state. (element >> 6)
    return tf.floor(tf.div(elementTensor, 64));
  }

  protected override mapTokensToTernaryFloats(tokens: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const tokensFloat = tf.cast(tokens, "float32");

      // Pure polynomial mapping requiring zero conditionals:
      // Token 0 -> 0.0
      // Token 1 -> 1.0
      // Token 2 -> -1.0
      const tSquared = tf.square(tokensFloat);
      const part1 = tf.mul(tSquared, -1.5);
      const part2 = tf.mul(tokensFloat, 2.5);

      return tf.add(part1, part2);
    });
  }

  protected override mutateElement(currentElement: number, gradientValue: number, learningRate: number): number {
    const sixBitState = currentElement & 0x3f;
    let weightToken = (currentElement >> 6) & 0x03;

    if (weightToken === 2) weightToken = -1;

    const result = this.optimizeSixBitElement(weightToken, sixBitState, gradientValue, learningRate);

    const validatedWeight = Math.max(-1, Math.min(1, result.newWeight));
    const validatedState = result.newSixBitState & 0x3f;

    const unsignedWeightToken = validatedWeight === -1 ? 2 : validatedWeight;

    return (unsignedWeightToken << 6) | validatedState;
  }

  protected abstract getInitialSixBitState(units: number, inFeatures: number): tf.Tensor;

  protected abstract optimizeSixBitElement(currentWeight: number, currentSixBitState: number, gradient: number, lr: number): { newWeight: number; newSixBitState: number };
}

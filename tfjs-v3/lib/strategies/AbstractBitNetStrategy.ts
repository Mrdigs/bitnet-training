import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "../IBitNetStrategy";

export abstract class AbstractBitNetStrategy implements IBitNetStrategy {
  protected bitsPerElement: number;
  protected elementsPerBlock: number;
  protected bitMask: number;
  protected strideMultipliers: number[];

  constructor(bitsPerElement: number, elementsPerBlock: number) {
    this.bitsPerElement = bitsPerElement;
    this.elementsPerBlock = elementsPerBlock;

    // Calculate the element's bitmask (e.g., 8 bits -> 255, 4 bits -> 15)
    this.bitMask = Math.pow(2, bitsPerElement) - 1;

    // Pre-calculate decimal multipliers for fast mathematical bit-shifting
    this.strideMultipliers = [];
    for (let i = 0; i < elementsPerBlock; i++) {
      this.strideMultipliers.push(Math.pow(2, i * bitsPerElement));
    }

    // Hardware compatibility guard: warn if configuration overflows WebGL float24 precision bounds
    const totalBitsUsed = bitsPerElement * elementsPerBlock;
    if (totalBitsUsed > 24) {
      console.warn(`[BitNet Warning]: Strategy uses ${totalBitsUsed} bits per block. ` + `This will run perfectly on Node.js/CPU, but will cause bit corruption if ported to standard browser WebGL backends.`);
    }
  }

  /**
   * Dynamically determines the packed shape based on elementsPerBlock.
   */
  public getPackedShape(units: number, inFeatures: number): [number, number] {
    if (inFeatures % this.elementsPerBlock !== 0) {
      throw new Error(`InFeatures (${inFeatures}) must be a multiple of elementsPerBlock (${this.elementsPerBlock}) ` + `for this configuration layout.`);
    }
    return [units, inFeatures / this.elementsPerBlock];
  }

  /**
   * Core initialization hook to transform float parameters into packed format.
   */
  public abstract prepareInitialWeights(rawFloatWeights: tf.Tensor2D): tf.Tensor;

  /**
   * Fully universal vectorized unpack loop for the forward pass (C++/GPU compatible).
   * Automatically scales its steps based on your configuration parameters.
   */
  public getTernaryWeights(packedTensor: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const segments: tf.Tensor[] = [];

      // Extract every packed channel dynamically using our pre-calculated stride multipliers
      for (let i = 0; i < this.elementsPerBlock; i++) {
        const divisor = this.strideMultipliers[i];
        const shifted = tf.floor(tf.div(packedTensor, divisor));
        const extractedSegment = tf.mod(shifted, this.bitMask + 1);

        // Let the child strategy decode this specific segment down to a ternary token
        segments.push(this.decodeElementToToken(extractedSegment));
      }

      const packedShape = packedTensor.shape;
      const units = packedShape[0];
      const uncompressedInFeatures = packedShape[1]! * this.elementsPerBlock;

      // Interleave and reshape the segments back into the uncompressed weight matrix [Units, InFeatures]
      const tokens = tf.reshape(tf.stack(segments, 2), [units, uncompressedInFeatures]);

      // Cast token indicators back to continuous float values (-1.0, 0.0, 1.0)
      return this.mapTokensToTernaryFloats(tokens);
    });
  }

  /**
   * Generalized optimization pass loop for the backward pass running on CPU memory.
   * Completely strips out all hardcoded offsets.
   */
  public applyGradientUpdate(currentPackedContainer: tf.Tensor, gradient: tf.Tensor, learningRate: number): tf.Tensor {
    const floatBuffer = currentPackedContainer.dataSync() as Float32Array;
    const intBuffer = new Int32Array(floatBuffer.buffer);
    const gradBuffer = gradient.dataSync() as Float32Array;

    const totalPackedElements = intBuffer.length;
    const updatedElementsLocal = new Array<number>(this.elementsPerBlock);

    // Step through each packed storage block element-by-element
    for (let i = 0; i < totalPackedElements; i++) {
      const packedVal = intBuffer[i];

      // 1. Unpack all elements residing inside this single storage block
      for (let e = 0; e < this.elementsPerBlock; e++) {
        const shiftBits = e * this.bitsPerElement;
        const rawElement = (packedVal >> shiftBits) & this.bitMask;

        // 2. Locate the corresponding coordinate inside the uncompressed gradient array
        const globalGradientIndex = i * this.elementsPerBlock + e;
        const gradScalar = gradBuffer[globalGradientIndex] ?? 0;

        // 3. Delegate the optimization step to the child strategy configuration
        const mutatedElement = this.mutateElement(rawElement, gradScalar, learningRate);
        updatedElementsLocal[e] = mutatedElement & this.bitMask;
      }

      // 4. Re-pack elements back into a single tracking register block
      let newPackedVal = 0;
      for (let e = 0; e < this.elementsPerBlock; e++) {
        newPackedVal |= updatedElementsLocal[e] << (e * this.bitsPerElement);
      }
      intBuffer[i] = newPackedVal;
    }

    return tf.tensor(floatBuffer, currentPackedContainer.shape, "float32");
  }

  // --- Abstract hooks to be satisfied by specific layout subclasses ---
  protected abstract decodeElementToToken(elementTensor: tf.Tensor): tf.Tensor;
  protected abstract mapTokensToTernaryFloats(tokens: tf.Tensor): tf.Tensor;
  protected abstract mutateElement(currentElement: number, gradientValue: number, learningRate: number): number;
}

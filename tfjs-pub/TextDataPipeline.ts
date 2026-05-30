import * as tf from "@tensorflow/tfjs";

export interface PipelineBatch {
  x: tf.Tensor2D; // Input token IDs matrix [batchSize, seqLen]
  y: tf.Tensor2D; // Target token IDs matrix [batchSize, seqLen] (shifted by 1)
}

export class TextDataPipeline {
  public readonly vocabSize: number;
  private charToId: Map<string, number> = new Map();
  private idToChar: Map<number, string> = new Map();

  // Contiguous VRAM storage for the entire corpus
  private datasetTensor: tf.Tensor1D;
  private datasetLength: number;

  constructor(rawText: string) {
    if (rawText.length === 0) {
      throw new Error("Cannot initialize data pipeline with an empty text string.");
    }

    // 1. Build Vocabulary Maps deterministically
    const uniqueChars = Array.from(new Set(rawText)).sort();
    this.vocabSize = uniqueChars.length;

    uniqueChars.forEach((char, index) => {
      this.charToId.set(char, index);
      this.idToChar.set(index, char);
    });

    // 2. Vectorize Entire Corpus into flat numeric array
    const encodedData = new Int32Array(rawText.length);
    for (let i = 0; i < rawText.length; i++) {
      encodedData[i] = this.charToId.get(rawText[i])!;
    }

    // 3. Persistent VRAM Allocation
    this.datasetLength = rawText.length;
    this.datasetTensor = tf.keep(tf.tensor1d(encodedData, "int32"));
  }

  /**
   * Public encoder helper for processing external inference text
   */
  public encode(text: string): number[] {
    const tokens: number[] = [];
    for (let i = 0; i < text.length; i++) {
      const id = this.charToId.get(text[i]);
      tokens.push(id !== undefined ? id : 0); // Default to token 0 for unknown chars
    }
    return tokens;
  }

  /**
   * Public decoder helper for text generation sampling loops
   */
  public decode(ids: number[]): string {
    return ids.map((id) => this.idToChar.get(id) || "").join("");
  }

  /**
   * Generates a 100% vectorized 2D autoregressive training batch entirely inside VRAM.
   * Implements Guru2's strict safety boundary buffer to completely eliminate shader crashes.
   */
  public nextBatch(batchSize: number, seqLen: number): PipelineBatch {
    return tf.tidy(() => {
      // Guru2's Safety Boundary Guard: Random start choices must allow room for full sequence + 1 target shift
      const maxValidIndex = this.datasetLength - seqLen - 1;
      if (maxValidIndex <= 0) {
        throw new Error(`Dataset text is too short for a context window of ${seqLen}.`);
      }

      // 1. Generate random starting offsets entirely on the CPU
      const startIndices = new Int32Array(batchSize);
      for (let b = 0; b < batchSize; b++) {
        startIndices[b] = Math.floor(Math.random() * maxValidIndex);
      }

      // 2. Build the 2D Sequential Gather Map Matrix [batchSize * seqLen]
      // Rather than looping tf.slice on the GPU, we expand the indices arithmetically
      const xFlattenedIndices = new Int32Array(batchSize * seqLen);
      const yFlattenedIndices = new Int32Array(batchSize * seqLen);

      for (let b = 0; b < batchSize; b++) {
        const baseIdx = startIndices[b];
        const batchOffset = b * seqLen;
        for (let s = 0; s < seqLen; s++) {
          xFlattenedIndices[batchOffset + s] = baseIdx + s;
          yFlattenedIndices[batchOffset + s] = baseIdx + s + 1; // Target is shifted exactly 1 token ahead
        }
      }

      // 3. Convert map arrays to 1D index tensors
      const xIndexTensor = tf.tensor1d(xFlattenedIndices, "int32");
      const yIndexTensor = tf.tensor1d(yFlattenedIndices, "int32");

      // 4. Run Vectorized Coalesced Gather Passes
      // Pulls every multi-dimensional character line in exactly one instruction block
      const xGathered = tf.gather(this.datasetTensor, xIndexTensor);
      const yGathered = tf.gather(this.datasetTensor, yIndexTensor);

      // 5. Reshape flat outputs back to [batchSize, seqLen] targets
      const x = xGathered.reshape([batchSize, seqLen]) as tf.Tensor2D;
      const y = yGathered.reshape([batchSize, seqLen]) as tf.Tensor2D;

      return { x, y };
    });
  }

  /**
   * Permanent resource teardown protocol
   */
  public dispose(): void {
    if (this.datasetTensor) {
      this.datasetTensor.dispose();
    }
  }
}

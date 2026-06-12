import * as tf from "@tensorflow/tfjs";

export class PersistentState {
  private tensors = new Map<string, tf.Tensor>();

  /**
   * Fetches an historical state tensor. Returns undefined if it hasn't been set yet.
   */
  public get(key: string): tf.Tensor | undefined {
    return this.tensors.get(key);
  }

  /**
   * Safe Getter/Initializer Combo: If the key doesn't exist, it executes your
   * factory function once, registers the result out of tidy bounds, and returns it.
   */
  public getOrCreate(key: string, initFactory: () => tf.Tensor): tf.Tensor {
    if (!this.tensors.has(key)) {
      const freshTensor = tf.keep(initFactory());
      this.tensors.set(key, freshTensor);
    }
    return this.tensors.get(key)!;
  }

  /**
   * Overwrites a tracking tensor, managing memory allocations safely.
   */
  public set(key: string, nextTensor: tf.Tensor): void {
    const oldTensor = this.tensors.get(key);

    // Retain the fresh tensor outside of transient execution scopes
    this.tensors.set(key, tf.keep(nextTensor));

    // Instantly reclaim memory from the previous step
    if (oldTensor) {
      oldTensor.dispose();
    }
  }

  public delete(key: string): void {
    const oldTensor = this.tensors.get(key);

    this.tensors.delete(key);

    if (oldTensor) {
      oldTensor.dispose();
    }
  }

  /**
   * Reclaims all GPU/CPU tracking memory allocated to this state instance.
   */
  public dispose(): void {
    this.tensors.forEach((tensor) => tensor.dispose());
    this.tensors.clear();
  }
}

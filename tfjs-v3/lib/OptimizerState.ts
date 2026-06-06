import * as tf from "@tensorflow/tfjs";

export class OptimizerState {
  private tensors = new Map<string, tf.Tensor>();

  public get(key: string): tf.Tensor | undefined {
    return this.tensors.get(key);
  }

  public set(key: string, nextTensor: tf.Tensor): void {
    const oldTensor = this.tensors.get(key);

    // FIX: Reference 'nextTensor' exactly to match the parameter name
    this.tensors.set(key, tf.keep(nextTensor));

    if (oldTensor) {
      oldTensor.dispose();
    }
  }

  public dispose(): void {
    this.tensors.forEach((tensor) => tensor.dispose());
    this.tensors.clear();
  }
}

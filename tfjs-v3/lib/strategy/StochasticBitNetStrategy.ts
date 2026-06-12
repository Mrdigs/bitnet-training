import * as tf from "@tensorflow/tfjs-node";
import { FourTo1BitPackingStrategy } from "./FourTo1BitPackingStrategy";
import { PersistentState } from "../PersistentState";

export class StochasticBitNetStrategy extends FourTo1BitPackingStrategy {
    
  public quantizeActivations(inputs: tf.Tensor, state: PersistentState): tf.Tensor {
    return inputs;
  }
  public dequantizeOutputs(rawOutputs: tf.Tensor, state: PersistentState): tf.Tensor {
    return rawOutputs;
  }

  public computeUpdate(weightTensor: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number): tf.Tensor {
    return tf.tidy(() => {
      const { weight, momentum } = this.unpack(weightTensor);

      // (Stochastic gradient optimization mechanics will live right here)

      return this.pack(weight, momentum);
    });
  }

  public applyUpdate(weightVar: tf.Variable, update: tf.Tensor): void {
    weightVar.assign(update);
  }
}

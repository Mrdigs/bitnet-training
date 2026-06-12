import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "../IBitNetStrategy";
import { PersistentState } from "../PersistentState";

export class ReferenceBitNetStrategy implements IBitNetStrategy {
  public getPackedShape(outFeatures: number, inFeatures: number): tf.Shape {
    return [inFeatures, outFeatures];
  }

  public prepareInitialWeights(rawFloatWeights: tf.Tensor): tf.Tensor {
    return rawFloatWeights.clone();
  }

  public decodeWeights(packedTensor: tf.Tensor, state: PersistentState): tf.Tensor {
    // Global per-tensor L1-norm scaling coefficient (Beta)
    const beta = tf.maximum(tf.mean(tf.abs(packedTensor)), tf.scalar(1e-5));
    state.set("beta", beta);

    return tf.customGrad((...args: any[]) => {
      const wIn = args[0] as tf.Tensor;
      const scaled = tf.div(wIn, beta);
      const ternary = tf.clipByValue(tf.round(scaled), -1, 1);
      return {
        value: ternary,
        gradFunc: (dy: tf.Tensor) => [dy],
      };
    })(packedTensor);
  }

  public quantizeActivations(inputs: tf.Tensor, state: PersistentState): tf.Tensor {
    const axis = inputs.rank - 1;

    // RMSNorm Phase
    const meanSquare = tf.mean(tf.square(inputs), axis, true);
    const rms = tf.sqrt(tf.add(meanSquare, tf.scalar(1e-5)));
    const xNorm = tf.div(inputs, rms);

    // Dynamic per-token scaling factor (Eta)
    const eta = tf.maximum(tf.max(tf.abs(xNorm), axis, true), tf.scalar(1e-5));
    state.set("eta", eta);

    return tf.customGrad((...args: any[]) => {
      const xIn = args[0] as tf.Tensor;
      const scaled = tf.mul(xIn, tf.div(tf.scalar(127.0, "float32"), eta));
      const quant = tf.clipByValue(tf.round(scaled), -127.0, 127.0);
      return {
        value: quant,
        gradFunc: (dy: tf.Tensor) => [dy],
      };
    })(xNorm);
  }

  public dequantizeOutputs(rawOutputs: tf.Tensor, state: PersistentState): tf.Tensor {
    const eta = state.get("eta");
    const beta = state.get("beta");

    if (!eta || !beta) {
      throw new Error("Dequantization failed: Context scales are missing.");
    }

    // FIX: Removed .dataSync() to keep the automatic differentiation tape intact
    return tf.mul(rawOutputs, tf.div(tf.mul(eta, beta), tf.scalar(127.0, "float32")));
  }

  public computeUpdate(weight: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number, currentStep: number): tf.Tensor {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    const firstMoment = state.getOrCreate("m", () => tf.zerosLike(weight));
    const secondMoment = state.getOrCreate("v", () => tf.zerosLike(weight));

    const nextM = tf.add(tf.mul(firstMoment, beta1), tf.mul(gradient, 1 - beta1));
    const nextV = tf.add(tf.mul(secondMoment, beta2), tf.mul(tf.square(gradient), 1 - beta2));

    // FIX: Use pure JS numeric calculations for step tracking to prevent graph memory leaks
    const mHat = tf.div(nextM, tf.scalar(1 - Math.pow(beta1, currentStep + 1)));
    const vHat = tf.div(nextV, tf.scalar(1 - Math.pow(beta2, currentStep + 1)));

    const updateDelta = tf.div(mHat, tf.add(tf.sqrt(vHat), tf.scalar(eps)));
    const nextWeight = tf.sub(weight, tf.mul(updateDelta, tf.scalar(learningRate)));

    state.set("m", nextM);
    state.set("v", nextV);

    return nextWeight;
  }

  public applyUpdate(weightVar: tf.Variable, update: tf.Tensor): void {
    weightVar.assign(update);
  }
}

import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "../IBitNetStrategy";
import { PersistentState } from "../PersistentState";
import { PackedBitNetStrategy } from "./PackedBitNetStrategy";

export class FusedStochasticBitNetStrategy extends PackedBitNetStrategy implements IBitNetStrategy {
  private readonly gradScale: number;
  private readonly K: number;

  private static readonly LOG_PROB_LUT_DATA: number[] = [1.0, 0.8, 0.64, 0.512, 0.41, 0.328, 0.262, 0.21, 0.168, 0.134, 0.107, 0.086, 0.069, 0.055, 0.044, 0.035, 0.028, 0.023, 0.018, 0.014, 0.012, 0.009, 0.007, 0.006, 0.005, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  private static readonly LUT_TENSOR: tf.Tensor1D = tf.tensor1d(FusedStochasticBitNetStrategy.LOG_PROB_LUT_DATA, "float32");

  public constructor(config?: { gradScale?: number; K?: number }) {
    super();
    this.gradScale = config?.gradScale ?? 10.0;
    this.K = config?.K ?? 0.5;
  }

  // -------------------------------------------------------------------------
  // Public Strategy API Methods
  // -------------------------------------------------------------------------

  public getPackedShape(outFeatures: number, inFeatures: number): tf.Shape {
    return [Math.ceil(inFeatures / 4), outFeatures];
  }

  public prepareInitialWeights(rawFloatWeights: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const signs = tf.sign(rawFloatWeights);
      const states = tf.add(signs, tf.scalar(1.0, "float32")) as tf.Tensor2D;

      const [inFeatures, outFeatures] = states.shape;
      const packedRows = Math.ceil(inFeatures / 4);

      const weightSlices: tf.Tensor2D[] = [];
      const zeroMomentumSlices: tf.Tensor2D[] = [];

      for (let i = 0; i < 4; i++) {
        if (i * packedRows < inFeatures) {
          const size = Math.min(packedRows, inFeatures - i * packedRows);
          let slice = tf.slice(states, [i * packedRows, 0], [size, outFeatures]);
          if (size < packedRows) {
            slice = tf.concat([slice, tf.ones([packedRows - size, outFeatures], "float32")], 0);
          }
          weightSlices.push(slice);
        } else {
          weightSlices.push(tf.ones([packedRows, outFeatures], "float32"));
        }
        zeroMomentumSlices.push(tf.zeros([packedRows, outFeatures], "float32"));
      }

      return this.pack(weightSlices, zeroMomentumSlices);
    });
  }

  public decodeWeights(packedTensor: tf.Tensor, state: PersistentState): tf.Tensor {
    return tf.tidy(() => {
      const subPackets = this.unpackBytes(packedTensor as tf.Tensor2D);

      const unpackedTernarySlices = subPackets.map((packet) => {
        const { weight } = this.decodeByteComponents(packet);
        return tf.sub(weight, tf.scalar(1.0, "float32"));
      });

      const fullWeightMatrix = tf.concat(unpackedTernarySlices, 0);
      const gamma = state.getOrCreate("gamma", () => tf.scalar(1.0, "float32"));
      return tf.mul(fullWeightMatrix, gamma);
    });
  }

  public quantizeActivations(inputs: tf.Tensor, state: PersistentState): tf.Tensor {
    const axis = inputs.rank - 1;
    const eta = tf.maximum(tf.max(tf.abs(inputs), axis, true), tf.scalar(1e-5));
    state.set("eta", eta);

    return tf.customGrad((...args: any[]) => {
      const xIn = args[0] as tf.Tensor;
      const scaled = tf.mul(xIn, tf.div(tf.scalar(127.0, "float32"), eta));
      const quant = tf.clipByValue(tf.round(scaled), -127.0, 127.0);
      return {
        value: quant,
        gradFunc: (dy: tf.Tensor) => [dy],
      };
    })(inputs);
  }

  public dequantizeOutputs(rawOutputs: tf.Tensor, state: PersistentState): tf.Tensor {
    const eta = state.get("eta");
    if (!eta) {
      throw new Error("Dequantization failed: Activation scale metadata 'eta' was missing.");
    }
    const etaVal = eta.dataSync();
    const staticEta = tf.tensor(etaVal, eta.shape, eta.dtype);
    const rescaled = tf.div(tf.mul(rawOutputs, staticEta), tf.scalar(127.0, "float32"));
    staticEta.dispose();
    return rescaled;
  }

  public computeUpdate(rawWeight: tf.Tensor, gradient: tf.Tensor, state: PersistentState, learningRate: number): tf.Tensor {
    return tf.tidy(() => {
      // FIXED: Remove .read(), treat weightVar directly as a tensor reference
      const fused = rawWeight as tf.Tensor2D;
      const fullGrads = gradient as tf.Tensor2D;
      const [packedRows, outFeatures] = fused.shape;

      // TODO: This ought to come from the argument
      const currentLrTensor = state.getOrCreate("lr", () => tf.scalar(0.001, "float32"));
      const scale_t = 1.0 + this.K * (1.0 - currentLrTensor.dataSync()[0]);

      // 1. Stack unpacked sub-bytes and slice gradients along Axis 0
      const packets = tf.stack(this.unpackBytes(fused), 0);
      const grads = tf.stack(this.sliceGradients(fullGrads, packedRows, outFeatures), 0);

      // 2. Extract weight and momentum components
      const weight = tf.mod(packets, tf.scalar(4.0, "float32"));
      const rawShifted = tf.floor(tf.div(packets, tf.scalar(4.0, "float32")));
      const unsignedMom = tf.mod(rawShifted, tf.scalar(64.0, "float32"));
      const isNegative = tf.greaterEqual(unsignedMom, tf.scalar(32.0, "float32"));
      const momentum = tf.where(isNegative, tf.sub(unsignedMom, tf.scalar(64.0, "float32")), unsignedMom);

      // 3. Parallelized Stochastic Momentum Step
      const momSign = tf.sign(momentum);
      const gradSign = tf.neg(tf.sign(grads));
      const activeAcceleration = tf.logicalAnd(tf.logicalOr(tf.equal(gradSign, momSign), tf.equal(momSign, 0)), tf.notEqual(gradSign, 0));

      const accelerationProbability = tf.clipByValue(tf.mul(tf.abs(grads), tf.scalar(this.gradScale, "float32")), 0.0, 1.0);
      const gatePassed = tf.less(tf.randomUniform(grads.shape, 0.0, 1.0, "float32"), accelerationProbability);
      const accelStep = tf.where(gatePassed, tf.scalar(1.0, "float32"), tf.scalar(0.0, "float32"));

      const finalStepDelta = tf.where(activeAcceleration, tf.mul(accelStep, gradSign), tf.mul(tf.scalar(-3.0, "float32"), momSign));
      const nextMomentum = tf.clipByValue(tf.add(momentum, finalStepDelta), -31.0, 31.0);

      // 4. Parallelized Annealed Weight Flip
      const annealedIndex = tf.clipByValue(tf.mul(tf.abs(nextMomentum), tf.scalar(scale_t, "float32")).toInt(), 0, 32);
      const flipProbability = tf.gather(FusedStochasticBitNetStrategy.LUT_TENSOR, annealedIndex);
      const shouldFlip = tf.less(tf.randomUniform(nextMomentum.shape, 0.0, 1.0, "float32"), flipProbability);

      const triggerUp = tf.logicalAnd(shouldFlip, tf.greater(nextMomentum, 0));
      const triggerDown = tf.logicalAnd(shouldFlip, tf.less(nextMomentum, 0));

      const weightIncrement = tf.where(tf.logicalAnd(triggerUp, tf.notEqual(weight, tf.scalar(2.0, "float32"))), tf.scalar(1.0, "float32"), tf.scalar(0.0, "float32"));
      const weightDecrement = tf.where(tf.logicalAnd(triggerDown, tf.notEqual(weight, tf.scalar(0.0, "float32"))), tf.scalar(1.0, "float32"), tf.scalar(0.0, "float32"));
      const updatedWeight = tf.add(tf.sub(weight, weightDecrement), weightIncrement);

      // 5. Parallelized Momentum Dampening
      const actualFlipOccurred = tf.greater(tf.add(weightIncrement, weightDecrement), 0);
      const dampedState = tf.where(tf.greater(nextMomentum, 0), tf.clipByValue(tf.sub(nextMomentum, 3.0), 0.0, 31.0), tf.clipByValue(tf.add(nextMomentum, 3.0), -31.0, 0.0));
      const dampedMomentum = tf.where(actualFlipOccurred, dampedState, nextMomentum);

      // 6. Restructure parallel tensor components back into isolated array structures
      const isMomNeg = tf.less(dampedMomentum, 0);
      const reUnsignedMomentum = tf.where(isMomNeg, tf.add(dampedMomentum, 64.0), dampedMomentum);
      const shiftedMomentum = tf.mul(reUnsignedMomentum, 4.0);
      const updatedBytePackets = tf.add(tf.clipByValue(updatedWeight, 0.0, 2.0), shiftedMomentum);

      const packetSlices = tf.unstack(updatedBytePackets, 0) as tf.Tensor2D[];
      const weightSlices = tf.unstack(updatedWeight, 0) as tf.Tensor2D[];

      // 7. Re-pack using the internal class helper method
      const nextPackedState = this.pack(packetSlices, tf.unstack(dampedMomentum, 0) as tf.Tensor2D[]);

      // 8. Re-evaluate global L1 Norm Variance Normalization Scalar (Gamma)
      const fullWeightMatrix = tf.concat(
        weightSlices.map((w) => tf.sub(w, tf.scalar(1.0, "float32"))),
        0,
      );

      const gammaData = tf.mean(tf.abs(fullWeightMatrix)).dataSync()[0];
      state.set("gamma", tf.scalar(gammaData > 0 ? gammaData : 1.0, "float32"));

      return nextPackedState;
    });
  }

  public applyUpdate(weightVar: tf.Variable, update: tf.Tensor): void {
    weightVar.assign(update);
  }
}

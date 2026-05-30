// layers/modularFusedBitNet/FusedBitNetLayer.ts
import * as tf from "@tensorflow/tfjs";
import { ParameterStorageCodec, ForwardProjectionQuantizer, InertiaUpdateStrategy, StochasticGateStrategy } from "./lib/interfaces";
import { DynamicSymmetricalCodec } from "./lib/codec/DynamicSymmetricalCodec";
import { BitNetGammaQuantizer } from "./lib/quantizer/BitNetGammaQuantizer";
import { ContinuousWarpInertiaStrategy } from "./lib/inertiaStrategy/ContinuousWarpInertiaStrategy";
import { TernaryStepGateStrategy } from "./lib/gateStrategy/TernaryStepGateStrategy";
import { Fused2bW6bMCodec } from "./lib/codec/Fused2bW6bMCodec";

export interface LayerConfig {
  inFeatures: number;
  outFeatures: number;
  gradScale?: number;
  K?: number;
  codec?: ParameterStorageCodec;
  quantizer?: ForwardProjectionQuantizer;
  inertiaStrategy?: InertiaUpdateStrategy;
  gateStrategy?: StochasticGateStrategy;
}

/**
 * FusedBitNetLayer
 *
 * Objective: provide a memory-efficient, 1-byte-per-parameter training
 * primitive for TensorFlow.js by fusing a ternary forward token and a
 * small signed momentum counter into a packed discrete state tensor.
 */
export class FusedBitNetLayer {
  public readonly inFeatures: number;
  public readonly outFeatures: number;
  public readonly gradScale: number;
  public readonly K: number;

  private packedState: tf.Tensor2D;

  // MODIFICATION: Set type signature to polymorphic 'any' to cleanly accept
  // either a flat JS primitive number or a vectorized column scale tf.Tensor.
  private currentScale: any = 1.0;

  private codec: ParameterStorageCodec;
  private quantizer: ForwardProjectionQuantizer;
  private inertiaStrategy: InertiaUpdateStrategy;
  private gateStrategy: StochasticGateStrategy;

  constructor(config: LayerConfig) {
    this.inFeatures = config.inFeatures;
    this.outFeatures = config.outFeatures;
    this.gradScale = config.gradScale ?? 30.0;
    this.K = config.K ?? 0.5;

    this.codec = config.codec ?? new Fused2bW6bMCodec();
    this.quantizer = config.quantizer ?? new BitNetGammaQuantizer();
    this.inertiaStrategy = config.inertiaStrategy ?? new ContinuousWarpInertiaStrategy(0.85);
    this.gateStrategy = config.gateStrategy ?? new TernaryStepGateStrategy();

    // 1. Allocate initial discrete state containers
    this.packedState = this.codec.getInitialState(this.inFeatures, this.outFeatures);

    // FIX: Extract initial weights and calculate the baseline scaling factor AT BIRTH
    // This forces Step 1 to execute under the exact variance-matched sigma ratio multiplier!
    tf.tidy(() => {
      const { weight } = this.codec.unpack(this.packedState);
      const rawScale = this.quantizer.calculateScale(weight);

      // MODIFICATION: If the quantizer outputs a blockwise tensor matrix, cache it inside VRAM memory lines safely
      this.currentScale = rawScale instanceof tf.Tensor ? tf.keep(rawScale) : rawScale;
    });
  }

  public forward(x: tf.Tensor2D): tf.Tensor2D {
    const codec = this.codec;
    const quantizer = this.quantizer;

    const customOp = tf.customGrad((...args: any[]): { value: tf.Tensor; gradFunc: (dy: tf.Tensor, saved: tf.Tensor[]) => tf.Tensor | tf.Tensor[] } => {
      const xInput = args[0] as tf.Tensor2D;
      const save = args[args.length - 1] as tf.GradSaveFunc;

      const fwdResult = tf.tidy(() => {
        const { weight } = codec.unpack(this.packedState);
        const realWeights = quantizer.transformWeights(weight);
        const output = tf.matMul(xInput, realWeights);

        // Element-wise broadcasting works natively whether currentScale is a number scalar or column tensor slice
        const scaledOutput = tf.mul(output, this.currentScale) as tf.Tensor2D;
        return { scaledOutput, realWeights };
      });

      save([fwdResult.realWeights]);

      const gradFunc = (dy: tf.Tensor, saved: tf.Tensor[]) => {
        return tf.tidy(() => {
          const [savedWeights] = saved;
          return tf.matMul(dy as tf.Tensor2D, (savedWeights as tf.Tensor2D).transpose()) as tf.Tensor2D;
        });
      };

      return { value: fwdResult.scaledOutput, gradFunc };
    });

    return customOp(x) as tf.Tensor2D;
  }

  public applyStep(gradients: tf.Tensor2D, currentLr: number): void {
    tf.tidy(() => {
      const scale_t = 1.0 + this.K * (1.0 - currentLr);

      const { weight, momentum } = this.codec.unpack(this.packedState);
      const nextMomentum = this.inertiaStrategy.update(momentum, gradients, this.gradScale);
      const { updatedWeight, dampedMomentum } = this.gateStrategy.evaluate(weight, nextMomentum, scale_t);
      const nextPackedState = this.codec.pack(updatedWeight, dampedMomentum);

      // MODIFICATION: Safely calculate the fresh blockwise column variance tensor maps
      const nextScale = this.quantizer.calculateScale(updatedWeight);

      // MODIFICATION: Track and dispose of stale scale textures to prevent absolute VRAM leaks
      const oldScale = this.currentScale;
      this.currentScale = nextScale instanceof tf.Tensor ? tf.keep(nextScale) : nextScale;
      if (oldScale instanceof tf.Tensor) oldScale.dispose();

      const oldPacked = this.packedState;
      this.packedState = tf.keep(nextPackedState);
      oldPacked.dispose();
    });
  }

  public dispose(): void {
    if (this.packedState) this.packedState.dispose();

    // MODIFICATION: Flush scale tensors out of the GPU backend at cleanup boundary passes
    if (this.currentScale instanceof tf.Tensor) {
      this.currentScale.dispose();
    }
  }
}

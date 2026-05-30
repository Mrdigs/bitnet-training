import * as tf from "@tensorflow/tfjs";

// Manages the literal bit layout and storage format inside VRAM
export interface ParameterStorageCodec {
  unpack(fused: tf.Tensor2D): { weight: tf.Tensor2D; momentum: tf.Tensor2D };
  pack(weight: tf.Tensor2D, momentum: tf.Tensor2D): tf.Tensor2D;
  getInitialState(inFeatures: number, outFeatures: number): tf.Tensor2D;
}

// Manages token-to-float translation and forward pass scale normalisation
export interface ForwardProjectionQuantizer {
  transformWeights(weightTokens: tf.Tensor2D): tf.Tensor2D;
  calculateScale(weightTokens: tf.Tensor2D): number | tf.Tensor;
}
// Evaluates continuous gradients to evolve the inertia registers
export interface InertiaUpdateStrategy {
  update(currentMomentum: tf.Tensor2D, gradients: tf.Tensor2D, gradScale: number): tf.Tensor2D;
}

// Triggers the stochastic weight state transitions and kinetic dampening
export interface StochasticGateStrategy {
  evaluate(
    weight: tf.Tensor2D,
    momentum: tf.Tensor2D,
    scale_t: number,
  ): {
    updatedWeight: tf.Tensor2D;
    dampedMomentum: tf.Tensor2D;
  };
}

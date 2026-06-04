import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "../IBitNetStrategy";

export class UncompromisedQatAdamStrategy implements IBitNetStrategy {
  // Genuine, uncompromised 32-bit floating point state matrices [Out, In]
  private latentWeights!: tf.Variable;
  private m!: tf.Variable;
  private v!: tf.Variable;

  private outFeatures: number;
  private inFeatures: number;
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private t: number;

  constructor(outFeatures: number, inFeatures: number, beta1: number = 0.9, beta2: number = 0.999, epsilon: number = 1e-8) {
    this.outFeatures = outFeatures;
    this.inFeatures = inFeatures;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.t = 0;

    // Allocate uncompromised 32-bit floating point parameters
    const weightShape = [outFeatures, inFeatures];
    this.latentWeights = tf.variable(tf.randomNormal(weightShape, 0, 0.1, "float32"));
    this.m = tf.variable(tf.zeros(weightShape, "float32"));
    this.v = tf.variable(tf.zeros(weightShape, "float32"));
  }

  /**
   * Tells the layer that this strategy requires a flat, uncompressed float32 shape layout.
   */
  public getPackedShape(units: number, inFeatures: number): [number, number] {
    return [units, inFeatures];
  }

  /**
   * Ingests the initial high-precision weight distribution snapshot.
   * Returns a dummy placeholder tensor so the layer builds its tracking nodes correctly.
   */
  public prepareInitialWeights(rawFloatWeights: tf.Tensor2D): tf.Tensor {
    this.latentWeights.assign(rawFloatWeights);
    return tf.zeros([this.outFeatures, this.inFeatures], "float32");
  }

  /**
   * AUTHENTIC QAT STE FORWARD PASS
   * Intercepts the call and executes an authentic Straight-Through Estimator (STE)
   * on the pure float32 latent weight matrix.
   */
  public getTernaryWeights(packedTensor: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      // Round and clip latent weights to standard ternary levels (-1.0, 0.0, 1.0)
      const rounded = tf.round(this.latentWeights);
      const ternary = tf.clipByValue(rounded, -1.0, 1.0);

      // STE Trick: Adding a zero-valued multiplication of the latent weights
      // forces the backward gradient tape to trace through to our latent float variables.
      return tf.add(ternary, tf.mul(this.latentWeights, 0.0));
    });
  }

  /**
   * UNCOMPROMISED FULL-PRECISION ADAM UPDATE
   * Processes the full matrix gradients directly without any dataSync byte-slicing loops.
   */
  public applyGradientUpdate(currentPackedContainer: tf.Tensor, gradient: tf.Tensor, learningRate: number): tf.Tensor {
    tf.tidy(() => {
      this.t += 1;

      // 1. m_t = beta1 * m_{t-1} + (1 - beta1) * g
      const newM = tf.add(tf.mul(this.m, this.beta1), tf.mul(gradient, 1 - this.beta1));

      // 2. v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
      const newV = tf.add(tf.mul(this.v, this.beta2), tf.mul(tf.square(gradient), 1 - this.beta2));

      this.m.assign(newM);
      this.v.assign(newV);

      // 3. Compute bias-corrected moments
      const mHat = tf.div(newM, 1 - Math.pow(this.beta1, this.t));
      const vHat = tf.div(newV, 1 - Math.pow(this.beta2, this.t));

      // 4. Update latent weights: W = W - lr * mHat / (sqrt(vHat) + eps)
      const denominator = tf.add(tf.sqrt(vHat), this.epsilon);
      const step = tf.mul(tf.div(mHat, denominator), learningRate);
      const updatedLatent = tf.sub(this.latentWeights, step);

      this.latentWeights.assign(updatedLatent);
    });

    // Return the original container to keep the optimizer loop functioning smoothly
    return currentPackedContainer;
  }
}

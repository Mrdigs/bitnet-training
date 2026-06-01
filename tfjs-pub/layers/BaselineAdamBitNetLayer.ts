// layers/BaselineAdamBitNetLayer.ts
import * as tf from "@tensorflow/tfjs";

export interface BaselineConfig {
  inFeatures: number;
  outFeatures: number;
  initialWeights?: tf.Tensor2D;
}

export class BaselineAdamBitNetLayer {
  public readonly inFeatures: number;
  public readonly outFeatures: number;

  private shadowWeights: tf.Tensor2D;
  private m: tf.Tensor2D;
  private v: tf.Tensor2D;
  private beta1 = 0.9;
  private beta2 = 0.999;
  private epsilon = 1e-8;
  private t = 0;

  constructor(config: BaselineConfig) {
    this.inFeatures = config.inFeatures;
    this.outFeatures = config.outFeatures;

    if (config.initialWeights) {
      this.shadowWeights = tf.keep(config.initialWeights.clone());
    } else {
      const scale = Math.sqrt(2.0 / (this.inFeatures + this.outFeatures));
      this.shadowWeights = tf.keep(tf.randomNormal([this.inFeatures, this.outFeatures], 0.0, scale, "float32"));
    }

    this.m = tf.keep(tf.zeros([this.inFeatures, this.outFeatures], "float32"));
    this.v = tf.keep(tf.zeros([this.inFeatures, this.outFeatures], "float32"));
  }

  public forward(x: tf.Tensor2D): tf.Tensor2D {
    return tf.tidy(() => {
      const meanVal = tf.mean(this.shadowWeights);
      const stdDev = tf.sqrt(tf.mean(tf.square(tf.sub(this.shadowWeights, meanVal))));
      const threshold = tf.mul(stdDev, tf.scalar(0.65, "float32"));

      const ternaryWeights = tf.where(tf.less(this.shadowWeights, tf.neg(threshold)), tf.scalar(-1.0, "float32"), tf.where(tf.greater(this.shadowWeights, threshold), tf.scalar(1.0, "float32"), tf.scalar(0.0, "float32")));

      const averageMagnitude = tf.mean(tf.abs(ternaryWeights));
      const gamma = tf.mul(averageMagnitude, stdDev);

      const output = tf.matMul(x, ternaryWeights);
      return tf.mul(output, gamma) as tf.Tensor2D;
    });
  }

  public applyStep(gradients: tf.Tensor2D, lr: number): void {
    tf.tidy(() => {
      this.t++;
      const safeGrads = gradients.clone();

      // FIX: Applied clear casting assignments across the intermediate Adam steps
      // to satisfy the type checker's Rank-2 constraint profiles natively.
      const nextM = tf.add(tf.mul(this.m, tf.scalar(this.beta1)), tf.mul(safeGrads, tf.scalar(1.0 - this.beta1))) as tf.Tensor2D;
      const nextV = tf.add(tf.mul(this.v, tf.scalar(this.beta2)), tf.mul(tf.square(safeGrads), tf.scalar(1.0 - this.beta2))) as tf.Tensor2D;

      const mHat = tf.div(nextM, tf.scalar(1.0 - Math.pow(this.beta1, this.t)));
      const vHat = tf.div(nextV, tf.scalar(1.0 - Math.pow(this.beta2, this.t)));

      const denom = tf.add(tf.sqrt(vHat), tf.scalar(this.epsilon));
      const stepUpdate = tf.mul(tf.div(mHat, denom), tf.scalar(lr));

      // FIX: Explicit intermediate type downcast prevents ts(2322) Rank errors
      const nextWeights: any = tf.sub(this.shadowWeights, stepUpdate);

      const oldW = this.shadowWeights;
      const oldM = this.m;
      const oldV = this.v;

      this.shadowWeights = tf.keep(nextWeights as tf.Tensor2D);
      this.m = tf.keep(nextM);
      this.v = tf.keep(nextV);

      oldW.dispose();
      oldM.dispose();
      oldV.dispose();
    });
  }

  public getWeights(): tf.Tensor2D {
    return tf.tidy(() => {
      const meanVal = tf.mean(this.shadowWeights);
      const stdDev = tf.sqrt(tf.mean(tf.square(tf.sub(this.shadowWeights, meanVal))));
      const threshold = tf.mul(stdDev, tf.scalar(0.65, "float32"));

      const ternaryWeights = tf.where(tf.less(this.shadowWeights, tf.neg(threshold)), tf.fill(this.shadowWeights.shape, -1.0), tf.where(tf.greater(this.shadowWeights, threshold), tf.fill(this.shadowWeights.shape, 1.0), tf.fill(this.shadowWeights.shape, 0.0))) as tf.Tensor2D;

      return ternaryWeights;
    });
  }

  public dispose(): void {
    if (this.shadowWeights) this.shadowWeights.dispose();
    if (this.m) this.m.dispose();
    if (this.v) this.v.dispose();
  }
}

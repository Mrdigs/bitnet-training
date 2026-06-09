import * as tf from "@tensorflow/tfjs";

export interface BitLinearLayerConfig {
  units: number;
  epsilon?: number;
  activation?: "relu" | "softmax" | "linear";
  inputShape?: tf.Shape;
}

export class ReferenceBitNetLayer extends tf.layers.Layer {
  public static className = "BitLinear";

  private readonly units: number;
  private readonly epsilon: number;
  private readonly activationType: "relu" | "softmax" | "linear";

  private weightMaster!: tf.LayerVariable;
  private rmsGamma!: tf.LayerVariable;

  constructor(config: BitLinearLayerConfig) {
    super(config as any);

    this.units = config.units;
    this.epsilon = config.epsilon ?? 1e-5;
    this.activationType = config.activation ?? "linear";

    this.supportsMasking = true;
  }

  public override build(inputShape: tf.Shape | tf.Shape[]): void {
    let shape: number[];

    if (Array.isArray(inputShape[0])) {
      const shapes = inputShape as number[][];
      if (shapes.length !== 1) {
        throw new Error(`BitLinear expects exactly one input shape, got ${shapes.length}`);
      }
      shape = shapes[0];
    } else {
      shape = inputShape as number[];
    }

    const inputDim = shape[shape.length - 1];

    if (inputDim === undefined || inputDim === null) {
      throw new Error("BitLinear requires a known input dimension.");
    }

    this.rmsGamma = this.addWeight("rms_gamma", [inputDim], "float32", tf.initializers.ones());
    this.weightMaster = this.addWeight("weight_master", [inputDim, this.units], "float32", tf.initializers.glorotUniform({}));

    this.built = true;
  }

  public override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      // 1. Input Parsing
      let x: tf.Tensor;
      if (Array.isArray(inputs)) {
        if (inputs.length !== 1) {
          throw new Error(`BitLinear expects exactly one input tensor, got ${inputs.length}`);
        }
        x = inputs[0];
      } else {
        x = inputs;
      }

      const axis = x.rank - 1;

      // 2. Pre-Layer Normalization (RMSNorm)
      const meanSquare = tf.mean(tf.square(x), axis, true);
      const rms = tf.sqrt(tf.add(meanSquare, tf.scalar(this.epsilon)));
      const xNorm = tf.mul(tf.div(x, rms), this.rmsGamma.read());

      // 3. Per-Token Activation Scale (η)
      const eta = tf.maximum(tf.max(tf.abs(xNorm), axis, true), tf.scalar(1e-5));

      // 4. Activation Quantization (INT8 [-127, 127]) via STE
      // Uses generic rest parameters to align perfectly with CustomGradientFunc signature
      const xQ = tf.customGrad((...args: any[]) => {
        const xIn = args[0] as tf.Tensor;
        const etaIn = args[1] as tf.Tensor;

        const scaled = tf.mul(xIn, tf.div(tf.scalar(127), etaIn));
        const quant = tf.clipByValue(tf.round(scaled), -127, 127);

        return {
          value: quant,
          gradFunc: (dy: tf.Tensor) => [dy, tf.zerosLike(etaIn)],
        };
      })(xNorm, eta);

      // 5. Per-Tensor Weight Scale (β)
      const wMasterTensor = this.weightMaster.read();
      const beta = tf.maximum(tf.mean(tf.abs(wMasterTensor)), tf.scalar(1e-5));

      // 6. Weight Quantization (Ternary {-1, 0, 1}) via STE
      // Uses generic rest parameters to align perfectly with CustomGradientFunc signature
      const wQ = tf.customGrad((...args: any[]) => {
        const wIn = args[0] as tf.Tensor;
        const betaIn = args[1] as tf.Tensor;

        const scaled = tf.div(wIn, betaIn);
        const ternary = tf.clipByValue(tf.round(scaled), -1, 1);

        return {
          value: ternary,
          gradFunc: (dy: tf.Tensor) => [dy, tf.zerosLike(betaIn)],
        };
      })(wMasterTensor, beta);

      // 7. Integer-Domain Matrix Multiplication
      const yInt = tf.matMul(xQ, wQ);

      // 8. Dequantization (Rescaling back into Float Domain)
      let y = tf.mul(yInt, tf.div(tf.mul(eta, beta), tf.scalar(127)));

      // 9. Post-Layer Dynamic Activation Application
      if (this.activationType === "relu") {
        y = tf.relu(y);
      } else if (this.activationType === "softmax") {
        y = tf.softmax(y);
      }

      return y;
    });
  }

  public override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    let shape: number[];

    if (Array.isArray(inputShape[0])) {
      const shapes = inputShape as number[][];
      shape = shapes[0];
    } else {
      shape = inputShape as number[];
    }

    const outputShape = [...shape];
    outputShape[outputShape.length - 1] = this.units;

    return outputShape;
  }

  public override getConfig(): tf.serialization.ConfigDict {
    return {
      ...super.getConfig(),
      units: this.units,
      epsilon: this.epsilon,
      activation: this.activationType,
    };
  }
}

tf.serialization.registerClass(ReferenceBitNetLayer);

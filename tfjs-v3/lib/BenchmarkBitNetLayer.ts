import * as tf from "@tensorflow/tfjs";

export interface BenchmarkLayerConfig {
  units: number;
  activation?: "relu" | "softmax" | "linear";
  inputShape?: tf.Shape;
}

export class BenchmarkBitNetLayer extends tf.layers.Layer {
  public static className = "BenchmarkBitNetLayer";

  private units: number;
  private activationName: "relu" | "softmax" | "linear";
  private kernelVar!: tf.LayerVariable;
  //private biasVar!: tf.LayerVariable;

  constructor(config: BenchmarkLayerConfig) {
    super(config as any);
    this.units = config.units;
    this.activationName = config.activation || "linear";
  }

  private unwrapShape(input: tf.Shape | tf.Shape[]): tf.Shape {
    return Array.isArray(input[0]) ? (input[0] as tf.Shape) : (input as tf.Shape);
  }

  private unwrapTensor(input: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return Array.isArray(input) ? input[0] : input;
  }

  public override build(inputShape: tf.Shape | tf.Shape[]): void {
    const singleShape = this.unwrapShape(inputShape);
    const inFeatures = singleShape[singleShape.length - 1];

    // Standard row-major weight allocation matching layer dimensions [Units, InFeatures]
    this.kernelVar = this.addWeight("kernel", [this.units, inFeatures], "float32", tf.initializers.glorotUniform({}));

    //this.biasVar = this.addWeight("bias", [this.units], "float32", tf.initializers.zeros());

    this.built = true;
  }

  public override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    const singleShape = this.unwrapShape(inputShape);
    const outputShape = [...singleShape];
    outputShape[outputShape.length - 1] = this.units;
    return outputShape;
  }

  public override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const inputTensor = this.unwrapTensor(inputs);
      const rawWeights = this.kernelVar.read();
      //const bias = this.biasVar.read();

      // THE BENCHMARK SHIELD: Emulates a native black-box hardware kernel
      const customGradFactory = tf.customGrad((...args: any[]) => {
        const x = args[0] as tf.Tensor;

        // 1. FORWARD PASS: Executed entirely off the tape using dataSync()
        const rawValues = x.dataSync();

        // Calculate the absolute mean scale factor directly on the CPU array values
        let sumAbsolute = 0;
        for (let i = 0; i < rawValues.length; i++) {
          sumAbsolute += Math.abs(rawValues[i]);
        }
        const scaleValue = sumAbsolute / rawValues.length + 1e-5;

        // Quantise continuous floating values to the nearest ternary integer bucket {-1, 0, 1}
        const decodedValues = new Float32Array(rawValues.length);
        for (let i = 0; i < rawValues.length; i++) {
          const normalized = rawValues[i] / scaleValue;
          decodedValues[i] = Math.max(-1.0, Math.min(1.0, Math.round(normalized)));
        }

        const ternaryOutputTensor = tf.tensor(decodedValues, x.shape, x.dtype);

        // 2. BACKWARD PASS: Strict proportional Straight-Through Estimator (STE)
        // Multiplying by the inverse forward scale factor ensures the gradients
        // arriving back at your continuous weights remain perfectly scaled.
        const gradFunc = (dy: tf.Tensor) => {
          const scaledGrad = tf.mul(dy, 1.0 / scaleValue);
          return [scaledGrad];
        };

        return { value: ternaryOutputTensor, gradFunc };
      });

      const executableTernaryWeights = customGradFactory(rawWeights);

      // Execute matrix multiplication using the aligned transposition flags
      const preActivation = tf.matMul(inputTensor, executableTernaryWeights); //, false, true);
      //const preActivation = tf.add(matrixProduct, bias);

      if (this.activationName === "relu") {
        return tf.relu(preActivation);
      } else if (this.activationName === "softmax") {
        return tf.softmax(preActivation, -1);
      }

      return preActivation;
    });
  }

  public override getConfig(): tf.serialization.ConfigDict {
    const config = super.getConfig();
    config.units = this.units;
    config.activation = this.activationName;
    return config;
  }

  public static override fromConfig<T extends tf.serialization.Serializable>(cls: tf.serialization.SerializableConstructor<T>, config: tf.serialization.ConfigDict): T {
    return new cls(config as any);
  }

  public override getClassName(): string {
    return BenchmarkBitNetLayer.className;
  }
}
tf.serialization.SerializationMap.register(BenchmarkBitNetLayer);

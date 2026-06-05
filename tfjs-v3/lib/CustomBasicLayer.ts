import * as tf from "@tensorflow/tfjs";

export interface CustomLayerArgs {
  units: number;
  activation?: "relu" | "softmax" | "linear";
  inputShape?: number[];
  name?: string;
}
export class CustomBasicDenseLayer extends tf.layers.Layer {
  public static className = "CustomBasicDenseLayer";

  private units: number;
  private activationName: "relu" | "softmax" | "linear";
  private kernelVar!: tf.LayerVariable;
  private biasVar!: tf.LayerVariable;

  constructor(args: CustomLayerArgs) {
    super(args as any);
    this.units = args.units;
    this.activationName = args.activation || "linear";
  }

  public override build(inputShape: tf.Shape | tf.Shape[]): void {
    const singleShape = Array.isArray(inputShape) ? (inputShape as tf.Shape) : (inputShape as tf.Shape);
    const inputDim = singleShape[singleShape.length - 1];

    this.kernelVar = this.addWeight("kernel", [inputDim, this.units], "float32", tf.initializers.glorotUniform({}));

    this.biasVar = this.addWeight("bias", [this.units], "float32", tf.initializers.zeros());

    this.built = true;
  }

  public override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    const singleShape = Array.isArray(inputShape) ? (inputShape as tf.Shape) : (inputShape as tf.Shape);
    const outputShape = [...singleShape];
    outputShape[outputShape.length - 1] = this.units;
    return outputShape;
  }

  public override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const inputTensor = Array.isArray(inputs) ? inputs[0] : inputs;

      const kernel = this.kernelVar.read();
      const bias = this.biasVar.read();

      const matrixProduct = tf.matMul(inputTensor, kernel);
      const preActivation = tf.add(matrixProduct, bias);

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
    return new cls(config);
  }

  public override getClassName(): string {
    return CustomBasicDenseLayer.className;
  }
}
tf.serialization.SerializationMap.register(CustomBasicDenseLayer);

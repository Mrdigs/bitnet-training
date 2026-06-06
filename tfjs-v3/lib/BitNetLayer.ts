import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "./IBitNetStrategy";

export interface BitNetLayerConfig {
  strategy: IBitNetStrategy;
  units: number;
  activation?: "relu" | "softmax" | "linear";
  inputShape?: tf.Shape; // Satisfies object creation signatures for starting layers
}

export class BitNetLayer extends tf.layers.Layer {
  public static className = "BitNetLayer";

  private strategy: IBitNetStrategy;
  private units: number;
  private activationName: "relu" | "softmax" | "linear";

  private kernelVar!: tf.LayerVariable;
  private biasVar!: tf.LayerVariable;

  constructor(config: BitNetLayerConfig) {
    // Framework pass-through: config captures inputShape natively here
    super(config as any);
    this.strategy = config.strategy;
    this.units = config.units;
    this.activationName = config.activation || "linear";
  }

  private unwrapShape(input: tf.Shape | tf.Shape[]): tf.Shape {
    return Array.isArray(input) ? (input as tf.Shape) : (input as tf.Shape);
  }

  private unwrapTensor(input: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return Array.isArray(input) ? input[0] : input;
  }

  public override build(inputShape: tf.Shape | tf.Shape[]): void {
    const singleShape = this.unwrapShape(inputShape);
    const inFeatures = singleShape[singleShape.length - 1]!;

    // Query strategy for its custom packing dimensions [Out, In]
    const packedShape = this.strategy.getPackedShape(this.units, inFeatures);

    this.kernelVar = this.addWeight("kernel", packedShape, "float32", tf.initializers.glorotUniform({}));

    this.biasVar = this.addWeight("bias", [this.units], "float32", tf.initializers.zeros());

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
      const rawPackedWeight = this.kernelVar.read();
      const bias = this.biasVar.read();

      // THE FRAMEWORK SHIELD: Protect the strategy from automatic tape lineage tracking
      const customGradFactory = tf.customGrad((...args: any[]) => {
        const x = args[0] as tf.Tensor;

        // Strategy Forward Pass: Execute your custom decoding routine (e.g. bitwise array transformations)
        const outputValue = this.strategy.decodeWeights(x);

        // Backward Pass: Hardcoded Straight-Through Estimator array signature
        const gradFunc = (dy: tf.Tensor) => [dy];

        return { value: outputValue, gradFunc };
      });

      const executableTernaryWeights = customGradFactory(rawPackedWeight);

      const matrixProduct = tf.matMul(inputTensor, executableTernaryWeights, false, true);
      const preActivation = tf.add(matrixProduct, bias);

      if (this.activationName === "relu") {
        return tf.relu(preActivation);
      } else if (this.activationName === "softmax") {
        return tf.softmax(preActivation, -1);
      }

      return preActivation;
    });
  }

  /**
   * Required for Serialization: Packages configuration states for model saves.
   */
  public override getConfig(): tf.serialization.ConfigDict {
    // super.getConfig() automatically serializes inputShape, batchInputShape, and names if they exist!
    const config = super.getConfig();
    config.units = this.units;
    config.activation = this.activationName;
    return config;
  }

  /**
   * Required for Deserialization: Reconstructs the layer instance from a saved configuration dictionary.
   */
  public static override fromConfig<T extends tf.serialization.Serializable>(cls: tf.serialization.SerializableConstructor<T>, config: tf.serialization.ConfigDict): T {
    return new cls(config as any);
  }

  public override getClassName(): string {
    return BitNetLayer.className;
  }
}
tf.serialization.SerializationMap.register(BitNetLayer);

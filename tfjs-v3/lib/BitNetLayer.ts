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

  /**
   * Corrected Shape Helper: Safely detects nested shape arrays by
   * checking if the first inner element is also an array.
   */
  private unwrapShape(input: tf.Shape | tf.Shape[]): tf.Shape {
    if (Array.isArray(input) && input.length > 0 && Array.isArray(input[0])) {
      return input[0] as tf.Shape; // Extract the first shape from a nested tf.Shape[]
    }
    return input as tf.Shape; // It's already a flat tf.Shape ([number, number])
  }

  /**
   * Corrected Tensor Helper: Checks if the first element is a true
   * Tensor instance to safely unwrap a collection.
   */
  private unwrapTensor(input: tf.Tensor | tf.Tensor[]): tf.Tensor {
    if (Array.isArray(input)) {
      return input[0]; // Extract the first tensor out of the array wrapper
    }
    return input; // Return the singular tensor directly
  }

  public override build(inputShape: tf.Shape | tf.Shape[]): void {
    console.log(`\n=====================================================`);
    console.log(`[SHAPE DIAGNOSTIC] BUILD LIFECYCLE FOR ${this.name}`);
    console.log(`=====================================================`);
    console.log(`  ▸ Raw Incoming inputShape:`, JSON.stringify(inputShape));

    const singleShape = this.unwrapShape(inputShape);
    console.log(`  ▸ Unwrapped Shape Output: `, JSON.stringify(singleShape));

    const inFeatures = singleShape[singleShape.length - 1];
    console.log(`  ▸ Extracted inFeatures Value:`, inFeatures);

    const packedShape = this.strategy.getPackedShape(this.units, inFeatures!);
    console.log(`  ▸ Strategy Requested Packed Weight Shape:`, JSON.stringify(packedShape));

    this.kernelVar = this.addWeight("kernel", packedShape, "float32", tf.initializers.glorotUniform({}));

    console.log(`  ▸ Allocated kernelVar True Shape:`, JSON.stringify(this.kernelVar.shape));

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

      const customGradFactory = tf.customGrad((...args: any[]) => {
        const x = args[0] as tf.Tensor;

        // Execute the strategy decoding pass
        const outputValue = this.strategy.decodeWeights(x);

        const gradFunc = (dy: tf.Tensor) => {
          return [dy];
        };

        return { value: outputValue, gradFunc };
      });

      const executableTernaryWeights = customGradFactory(rawPackedWeight);
      const matrixProduct = tf.matMul(inputTensor, executableTernaryWeights);
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

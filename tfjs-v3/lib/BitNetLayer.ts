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

  public override build(inputShape: tf.Shape | tf.Shape[]): void {
    // 1. Properly detect if the framework is delivering a collection of distinct branch shapes
    // Check if the first inner item is an array (indicating inputShape is a true tf.Shape[])
    const isMultiInput = Array.isArray(inputShape) && inputShape.length > 0 && Array.isArray(inputShape[0]);

    // 2. Standardise into a predictable array of shapes so our code handles both paths identically
    const allInputShapes: tf.Shape[] = isMultiInput ? (inputShape as tf.Shape[]) : [inputShape as tf.Shape];

    // 3. Extract the feature count of your primary data matrix entry (the first branch)
    const primaryShape = allInputShapes[0];
    const inFeatures = primaryShape[primaryShape.length - 1]!;

    // 4. Query your strategy for its custom packing setup dimensions
    const packedShape = this.strategy.getPackedShape(this.units, inFeatures);

    this.kernelVar = this.addWeight("kernel", packedShape, "float32", tf.initializers.glorotUniform({}));

    this.biasVar = this.addWeight("bias", [this.units], "float32", tf.initializers.zeros());

    this.built = true;
  }

  public override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    // 1. Detect if the incoming metadata configuration is multi-input or single-channel
    const isMultiInput = Array.isArray(inputShape) && inputShape.length > 0 && Array.isArray(inputShape[0]);

    // 2. Extract the primary tensor shape block that drives the matrix multiplication
    const primaryShape = isMultiInput ? (inputShape as tf.Shape[])[0] : (inputShape as tf.Shape);

    // 3. Clone the primary spatial bounds (preserving dynamic batch tokens like 'null' or batch sizes)
    const outputShape = [...primaryShape];

    // 4. Mutate ONLY the trailing feature channel axis to match your layer's target units count
    outputShape[outputShape.length - 1] = this.units;

    return outputShape;
  }

  /**
   * Helper: Normalizes dynamic tensor arguments down to a reliable flat array list.
   */
  private normalizeTensors(input: tf.Tensor | tf.Tensor[]): tf.Tensor[] {
    return Array.isArray(input) ? input : [input];
  }

  public override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      // 1. Safe normalization: Protects parallel inputs instead of throwing them away
      const allInputTensors = this.normalizeTensors(inputs);

      // 2. Extract the primary driving feature matrix (the first tensor stream)
      const primaryInputTensor = allInputTensors[0];

      const rawPackedWeight = this.kernelVar.read();
      const bias = this.biasVar.read();

      // THE FRAMEWORK SHIELD: Protect the strategy from automatic tape lineage tracking
      const customGradFactory = tf.customGrad((...args: any[]) => {
        const x = args[0] as tf.Tensor;

        // Execute your strategy's forward decoding calculation (black-box math execution)
        const outputValue = this.strategy.decodeWeights(x);

        // Backward Pass: Hardcoded Straight-Through Estimator array signature
        const gradFunc = (dy: tf.Tensor) => [dy];

        return { value: outputValue, gradFunc };
      });

      const executableTernaryWeights = customGradFactory(rawPackedWeight);

      // Execute matrix multiplication using the unpacked weights
      const matrixProduct = tf.matMul(primaryInputTensor, executableTernaryWeights);

      // Apply the standard bias offset
      const preActivation = tf.add(matrixProduct, bias);

      // Compute the framework layer activations natively
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

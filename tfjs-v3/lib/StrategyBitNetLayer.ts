import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "./IBitNetStrategy";
import { PersistentState } from "./PersistentState";

export interface BitNetLayerConfig {
  strategy: IBitNetStrategy;
  units: number;
  activation?: "relu" | "softmax" | "linear";
  inputShape?: tf.Shape; // Satisfies object creation signatures for starting layers
}

export class StrategyBitNetLayer extends tf.layers.Layer {
  public static className = "StrategyBitNetLayer";

  private strategy: IBitNetStrategy;
  private units: number;
  private activationName: "relu" | "softmax" | "linear";

  private kernelVar!: tf.LayerVariable;

  private layerState = new PersistentState();

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

      // 1. Pass the layer's dedicated memory state straight to your strategy activation step
      const quantizedInputs = this.strategy.quantizeActivations(primaryInputTensor, this.layerState);

      // 2. Protect weight matrix tape boundaries
      const customGradFactory = tf.customGrad((...args: any[]) => {
        const x = args[0] as tf.Tensor;
        const outputValue = this.strategy.decodeWeights(x);
        const gradFunc = (dy: tf.Tensor) => [dy];
        return { value: outputValue, gradFunc };
      });

      const executableTernaryWeights = customGradFactory(rawPackedWeight);

      // 3. Forward pass matrix multiplication execution
      const preActivation = tf.matMul(quantizedInputs, executableTernaryWeights);

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
    return StrategyBitNetLayer.className;
  }
}
tf.serialization.SerializationMap.register(StrategyBitNetLayer);

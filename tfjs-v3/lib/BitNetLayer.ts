import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "./IBitNetStrategy";

export interface BitNetLayerConfig {
  strategy: IBitNetStrategy;
  units: number;
  inputShape?: tf.Shape;
  weights?: tf.Tensor[];
}

export class BitNetLayer extends tf.layers.Layer {
  private config: BitNetLayerConfig;
  declare private kernel: tf.LayerVariable;

  constructor(config: BitNetLayerConfig) {
    const { weights, ...baseConfig } = config;
    super(baseConfig as unknown as tf.serialization.ConfigDict);
    this.config = config;
  }

  public override build(inputShape: tf.Shape | tf.Shape[]): void {
    // Standardize input unwrap: Extract the first flat shape vector cleanly
    const flatInputShape = (Array.isArray(inputShape[0]) ? inputShape[0] : inputShape) as tf.Shape;

    if (!flatInputShape || !Array.isArray(flatInputShape)) {
      throw new Error("BitNetLayer: Could not determine input shape during build sequence.");
    }

    const inFeatures = flatInputShape[flatInputShape.length - 1];
    if (inFeatures === undefined || inFeatures === null) {
      throw new Error("BitNetLayer requires a known, non-null input feature dimension shape.");
    }

    const containerShape = this.config.strategy.getPackedShape(this.config.units, inFeatures);
    this.kernel = this.addWeight("kernel", containerShape, "float32", tf.initializers.zeros(), undefined, true);

    if (this.config.weights && this.config.weights.length > 0) {
      tf.tidy(() => {
        const rawInitialMatrix = this.config.weights![0] as tf.Tensor2D;
        const formattedWeights = this.config.strategy.prepareInitialWeights(rawInitialMatrix);
        this.kernel.write(formattedWeights);
      });
    }
  }

  public override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const inputTensor = Array.isArray(inputs) ? inputs[0] : inputs;
      const rawTensor = this.kernel.read();
      const ternaryWeights = this.config.strategy.getTernaryWeights(rawTensor);
      return tf.matMul(inputTensor, ternaryWeights, false, true);
    });
  }

  public override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape {
    // FIX: Match the exact working extraction index structure of the build method
    // If inputShape is [[null, 784]], this unwraps it perfectly down to a raw iterable array: [null, 784]
    const flatInputShape = (Array.isArray(inputShape[0]) ? inputShape[0] : inputShape) as tf.Shape;

    if (!flatInputShape || !Array.isArray(flatInputShape)) {
      throw new Error("BitNetLayer: Invalid inputShape format inside computeOutputShape.");
    }

    // This spread operation is now guaranteed a flat, standard iterable array of numbers
    const outputShape = [...flatInputShape];
    outputShape[outputShape.length - 1] = this.config.units;

    return outputShape;
  }

  public override getWeights(): tf.Tensor[] {
    return [this.kernel.read()];
  }

  static get className(): string {
    return "BitNetLayer";
  }
}

tf.serialization.registerClass(BitNetLayer);

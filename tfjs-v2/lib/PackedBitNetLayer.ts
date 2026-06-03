import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "./strategy/IBitNetStrategy";

export interface PackedBitNetLayerConfig {
  units: number;
  strategy: IBitNetStrategy;
  activation?: string;
  customInitialWeights?: tf.Tensor;
  name?: string;
  trainable?: boolean;
  inputShape?: tf.Shape;
}

export class PackedBitNetLayer extends tf.layers.Layer {
  private units: number;
  private activation: (x: tf.Tensor) => tf.Tensor;
  private strategy: IBitNetStrategy;
  private customInitialWeights?: tf.Tensor;
  public packedKernel!: tf.LayerVariable;

  constructor(config: PackedBitNetLayerConfig) {
    super(config as any);
    this.units = config.units;
    this.strategy = config.strategy;
    this.customInitialWeights = config.customInitialWeights;
    this.activation = config.activation === "relu" ? tf.relu : (x) => x;
  }

  private normalizeShape(shape: tf.Shape | tf.Shape[]): tf.Shape {
    if (Array.isArray(shape[0])) {
      return (shape as tf.Shape[])[0];
    }
    return shape as tf.Shape;
  }

  public build(inputShape: tf.Shape | tf.Shape[]): void {
    const cleanShape = this.normalizeShape(inputShape);
    const inputDim = cleanShape[cleanShape.length - 1]!;
    const weightShape = [inputDim, this.units];

    let packedInitialValues: tf.Tensor;

    if (this.customInitialWeights) {
      packedInitialValues = this.strategy.prepareInitialWeights(this.customInitialWeights);
    } else {
      const rawRandom = tf.randomUniform(weightShape, -1, 1);
      packedInitialValues = this.strategy.prepareInitialWeights(rawRandom);
    }

    // Lie about data type metadata to trigger native autograd tape tracking nodes,
    // but initialize with a raw int32 tensor to dictate a compact integer RAM footprint.
    this.packedKernel = this.addWeight(
      "packedKernel",
      weightShape,
      "float32" as any,
      {
        apply: () => tf.clone(packedInitialValues),
        getClassName: () => "CustomTensorInitializer",
      } as any,
    );

    // FIX: Extract the raw underlying tf.Variable instance wrapper and
    // register its path out-of-band inside the optimizer's static lookup dictionary
    //const rawVariableInstance = (this.packedKernel as any).val as tf.Variable;
    //PackedBitNetOptimizer.registerKernel(this.packedKernel.name, rawVariableInstance);

    this.built = true;
  }

  public computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    const shape = [...this.normalizeShape(inputShape)];
    shape[shape.length - 1] = this.units;
    return shape;
  }

  public call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const inputTensor = Array.isArray(inputs) ? (inputs[0] as tf.Tensor) : (inputs as tf.Tensor);
      const packedVarTensor = this.packedKernel.read();

      // Sever the tape from the non-differentiable unpack logic operations
      const detachedPackedData = tf.tensor(packedVarTensor.dataSync(), packedVarTensor.shape, "int32");
      const { ternaryWeights: rawFloatWeights } = this.strategy.unpack(detachedPackedData);

      // Connect the active forward graph directly to the packedKernel tracking node using customGrad
      const tapeBridge = tf.customGrad((...args: any[]) => {
        const gradFunc = (dy: tf.Tensor) => [dy];
        return { value: rawFloatWeights, gradFunc };
      });

      const floatWeightsForGraph = tapeBridge(packedVarTensor);

      const linearOutput = tf.matMul(inputTensor, floatWeightsForGraph);
      return this.activation(linearOutput);
    });
  }

  static get className(): string {
    return "PackedBitNetLayer";
  }
}

tf.serialization.registerClass(PackedBitNetLayer);

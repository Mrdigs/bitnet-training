import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "./IBitNetStrategy";
import { PersistentState } from "./PersistentState";
import { ILearningRate } from "./LearningRate";

export interface StrategyBitNetLayerConfig {
  strategy: IBitNetStrategy;
  learningRate: ILearningRate;
  units: number;
  activation?: "relu" | "softmax" | "linear";
  inputShape?: tf.Shape; // Satisfies object creation signatures for starting layers
}

export class StrategyBitNetLayer extends tf.layers.Layer {
  public static className = "StrategyBitNetLayer";

  private readonly units: number;
  private readonly activationType: "relu" | "softmax" | "linear";

  private readonly strategy: IBitNetStrategy;
  private readonly learningRate: ILearningRate;
  private readonly layerState: PersistentState;

  private kernelVar!: tf.LayerVariable;

  constructor(config: StrategyBitNetLayerConfig) {
    super(config as any);
    this.units = config.units;
    this.strategy = config.strategy;
    this.activationType = config.activation ?? "linear";
    this.learningRate = config.learningRate;
    this.layerState = new PersistentState();
    // this.supportsMasking = true;
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
      throw new Error("StrategyBitNetLayer requires a known input dimension.");
    }

    const packedShape = this.strategy.getPackedShape(this.units, inputDim);

    this.kernelVar = this.addWeight(
      "kernel",
      packedShape,
      "float32", // Statically enforced to preserve framework autodiff trainability pipelines
      tf.initializers.zeros(),
    );

    // 2. Wrap initialization logic inside tidy to prevent memory accumulation on the GPU texture tape
    tf.tidy(() => {
      // 3. Generate high-fidelity standard Glorot Uniform weights matching uncompressed [in, out] dimensions
      const floatInitializer = tf.initializers.glorotUniform({});
      const baseFloatWeights = floatInitializer.apply([inputDim, this.units], "float32");

      // 4. Transform the floating-point initialization values into the strategy's target configuration
      const transformedInitialWeights = this.strategy.prepareInitialWeights(baseFloatWeights, this.layerState);

      // 5. Commit an atomic assignment operation to push the packed parameter configuration directly to the GPU
      // Read out the target underlying variable reference and force mutate it.
      (this.kernelVar as any).write(transformedInitialWeights);
    });

    this.built = true;
  }

  public override call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      // 1. Input Parsing
      let x: tf.Tensor;
      if (Array.isArray(inputs)) {
        if (inputs.length !== 1) {
          throw new Error(`Layer expects exactly one input tensor, got ${inputs.length}`);
        }
        x = inputs[0];
      } else {
        x = inputs;
      }
      const rawPackedWeight = this.kernelVar.read();

      // 1. Transform active input activations via strategy
      const processedInputs = this.strategy.quantizeActivations(x, this.layerState);

      // 2. Straight-Through Estimator weight decoder mapping hook
      const customGradFactory = tf.customGrad((...args: any[]) => {
        const wIn = args[0] as tf.Tensor;
        const decoded = this.strategy.decodeWeights(wIn, this.layerState);
        return {
          value: decoded,
          gradFunc: (dy: tf.Tensor) => {
            return tf.tidy(() => {
              const currentStep = this.learningRate.getStep();
              const learningRate = this.learningRate.getLearningRate();
              return [this.strategy.computeUpdate(rawPackedWeight, dy, this.layerState, learningRate, currentStep)];
            });
          },
        };
      });
      const executableWeights = customGradFactory(rawPackedWeight);

      // 3. Perform integer domain Matrix Multiplication
      const matMulOutputs = tf.matMul(processedInputs, executableWeights);

      // 4. Rescale output metrics via strategy dequantization pass
      const preActivation = this.strategy.dequantizeOutputs(matMulOutputs, this.layerState);

      // 5. Post-layer dynamic activation routing
      if (this.activationType === "relu") {
        return tf.relu(preActivation);
      } else if (this.activationType === "softmax") {
        return tf.softmax(preActivation, -1);
      }
      return preActivation;
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
      activation: this.activationType,
    };
  }
}

tf.serialization.registerClass(StrategyBitNetLayer);

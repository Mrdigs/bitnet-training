import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "./strategy/IBitNetStrategy";

export class PackedBitNetOptimizer extends tf.Optimizer {
  constructor(
    private strategy: IBitNetStrategy,
    private learningRate = 0.001,
  ) {
    super();
    this.applyGradients = this.applyGradients.bind(this);
  }

  public getConfig(): tf.serialization.ConfigDict {
    return {
      learningRate: this.learningRate,
      strategyClassName: (this.strategy.constructor as any).className || "CustomBitNetStrategy",
    };
  }

  public applyGradients(variableGradients: tf.NamedTensorMap | any[]): void {
    //console.log("CAlled with", variableGradients);
    tf.tidy(() => {
      const gradients = Array.isArray(variableGradients)
        ? variableGradients.map((g: any) => ({ name: g.name, grad: g.grad || g.tensor }))
        : Object.keys(variableGradients).map((name) => ({
            name,
            grad: (variableGradients as any)[name].grad,
          }));

      const engineVars = tf.engine().registeredVariables as any;

      for (const { name, grad } of gradients) {
        if (!name || !grad) continue;

        const trueVariableRef = engineVars[name] as tf.Variable;

        if (trueVariableRef) {
          // Cast the tensor object directly to int32 to safely execute your strategy math
          const currentPackedData = tf.cast(trueVariableRef, "int32");

          const { velocity, ternaryWeights } = this.strategy.unpack(currentPackedData);

          const { finalVelocity, finalTernaryWeights } = this.strategy.mutate(
            velocity,
            ternaryWeights,
            grad, // Now receives valid multi-layer error derivatives (X^T · dy)
            this.learningRate,
          );

          const newPackedData = this.strategy.pack(finalVelocity, finalTernaryWeights);

          // Write updates back into the hardware array container variable
          trueVariableRef.assign(tf.cast(newPackedData, "float32"));
        }
      }
    });
  }

  public initOptimizerVars(): void {}
  public getClassName(): string {
    return "PackedBitNetOptimizer";
  }
}

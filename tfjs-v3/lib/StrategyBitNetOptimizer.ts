import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "./IBitNetStrategy";

export class StrategyBitNetOptimizer extends tf.Optimizer {
  public static className = "StrategyBitNetOptimizer";

  private strategy: IBitNetStrategy;

  constructor(strategy: IBitNetStrategy) {
    super();
    this.strategy = strategy;
  }

  public override applyGradients(variableGradientsMap: any[]): void {
    tf.tidy(() => {
      const registeredVars = tf.engine().registeredVariables;

      for (const item of variableGradientsMap) {
        const varName = item.name;
        const grad = item.tensor;
        const liveWeightVar = registeredVars[varName] as tf.Variable;

        if (!liveWeightVar || grad == null) {
          continue;
        }

        if (varName.includes("kernel")) {
          this.strategy.applyUpdate(liveWeightVar, grad);
        } else {
          // Biases or background parameters
          const delta = grad.mul(0.01);
          liveWeightVar.assign(liveWeightVar.sub(delta));
        }
      }
    });
  }

  public override getConfig(): tf.serialization.ConfigDict {
    return {};
  }
}
tf.serialization.SerializationMap.register(StrategyBitNetOptimizer);

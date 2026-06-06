import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "./IBitNetStrategy";
import { OptimizerState } from "./OptimizerState";

export class BitNetOptimizer extends tf.Optimizer {
  public static className = "BitNetOptimizer";

  private strategy: IBitNetStrategy;
  private states = new Map<string, OptimizerState>();

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
          if (!this.states.has(varName)) {
            this.states.set(varName, new OptimizerState());
          }
          const stateContainer = this.states.get(varName)!;

          const maxBefore = liveWeightVar.max().dataSync()[0];
          const minBefore = liveWeightVar.min().dataSync()[0];

          // Fire your custom strategy update logic
          this.strategy.computeUpdate(liveWeightVar, grad, stateContainer);
        } else {
          // Biases or background parameters
          const delta = grad.mul(0.01);
          liveWeightVar.assign(liveWeightVar.sub(delta));
        }
      }
    });
  }

  public override dispose(): void {
    super.dispose();
    this.states.forEach((stateContainer) => stateContainer.dispose());
    this.states.clear();
  }

  public override getConfig(): tf.serialization.ConfigDict {
    return {};
  }
}
tf.serialization.SerializationMap.register(BitNetOptimizer);

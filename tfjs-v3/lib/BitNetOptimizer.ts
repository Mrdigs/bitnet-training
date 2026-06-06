import * as tf from "@tensorflow/tfjs";
import { IBitNetStrategy } from "./IBitNetStrategy";
import { OptimizerState } from "./OptimizerState";

export class BitNetOptimizer extends tf.Optimizer {
  public static className = "BitNetOptimizer";

  private strategy: IBitNetStrategy;
  // Simplest storage structure: map string variable names straight to concrete state containers
  private states = new Map<string, OptimizerState>();

  constructor(strategy: IBitNetStrategy) {
    super();
    this.strategy = strategy;
  }

  public override applyGradients(variableGradientsMap: any[]): void {
    tf.tidy(() => {
      const registeredVars = tf.engine().registeredVariables;

      for (const item of variableGradientsMap) {
        if (!item || item.grad == null) continue;

        const varName = item.name;
        const grad = item.tensor as tf.Tensor;
        const liveWeightVar = registeredVars[varName] as tf.Variable;

        if (!liveWeightVar) continue;

        if (varName.includes("kernel")) {
          // 1. Lazy-load a clean, concrete state instance for this specific weight variable
          if (!this.states.has(varName)) {
            this.states.set(varName, new OptimizerState());
          }
          const stateContainer = this.states.get(varName)!;

          // 2. Delegate the calculation entirely to your pure mathematical strategy code
          this.strategy.computeUpdate(liveWeightVar, grad, stateContainer);
        } else {
          // Hardened baseline backup fallback step size for unmanaged nodes (like biases)
          const delta = grad.mul(0.01);
          liveWeightVar.assign(liveWeightVar.sub(delta));
        }
      }
    });
  }

  public override dispose(): void {
    super.dispose();
    // Simply tell each individual container instance to clear out its own memory footprints
    this.states.forEach((stateContainer) => stateContainer.dispose());
    this.states.clear();
  }

  public override getConfig(): tf.serialization.ConfigDict {
    return {};
  }
}
tf.serialization.SerializationMap.register(BitNetOptimizer);

import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "./IBitNetStrategy";

export class BitNetOptimizer {
  private strategy: IBitNetStrategy;
  private learningRate: number;

  constructor(strategy: IBitNetStrategy, learningRate: number = 0.001) {
    this.strategy = strategy;
    this.learningRate = learningRate;
  }

  /**
   * Evaluates backward gradients and maps their target updates directly
   * onto layer allocations via strategy dependency injection.
   */
  public applyGradients(variableGradients: tf.NamedTensorMap): void {
    tf.tidy(() => {
      // Query active structural memory addresses tracking our models weights
      const engineVars = tf.engine().registeredVariables as Record<string, tf.Variable>;

      for (const name of Object.keys(variableGradients)) {
        const trueVariableRef = engineVars[name];
        if (!trueVariableRef) continue;

        const currentGrad = variableGradients[name];
        if (!currentGrad) continue;

        // Hand complete control over to the strategy to calculate mutations
        const updatedContainer = this.strategy.applyGradientUpdate(trueVariableRef, currentGrad, this.learningRate);

        // Commit updates instantly inside the native C++ runtime heap allocation
        trueVariableRef.assign(updatedContainer);
      }
    });
  }
}

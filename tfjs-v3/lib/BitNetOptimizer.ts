import * as tf from "@tensorflow/tfjs-node";
import { IBitNetStrategy } from "./IBitNetStrategy";

export class BitNetOptimizer {
  private strategy: IBitNetStrategy;
  private learningRate: number;

  constructor(strategy: IBitNetStrategy, learningRate: number = 0.001) {
    this.strategy = strategy;
    this.learningRate = learningRate;
  }

  public applyGradients(variableGradients: tf.NamedTensorMap): void {
    tf.tidy(() => {
      const engineVars = tf.engine().registeredVariables as Record<string, tf.Variable>;

      for (const name of Object.keys(variableGradients)) {
        const trueVariableRef = engineVars[name];
        if (!trueVariableRef) continue;

        const currentGrad = variableGradients[name];
        if (!currentGrad) continue;

        // Pass the raw live variable and its matching gradient straight to the strategy.
        const updatedContainer = this.strategy.applyGradientUpdate(trueVariableRef, currentGrad, this.learningRate);

        // Commit the updated tensor container cleanly back onto the engine variable memory heap
        trueVariableRef.assign(updatedContainer);
      }
    });
  }
}

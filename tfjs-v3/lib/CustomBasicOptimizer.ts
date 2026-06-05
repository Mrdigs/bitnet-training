import * as tf from "@tensorflow/tfjs";

interface CustomOptimizerConfig extends tf.serialization.ConfigDict {
  learningRate: number;
}

export class CustomBasicOptimizer extends tf.Optimizer {
  public static className = "CustomBasicOptimizer";

  private learningRate: number;
  private stateMap: Map<string, tf.Tensor>;
  private debugStepCount = 0;

  constructor(learningRate = 0.01) {
    super();
    this.learningRate = learningRate;
    this.stateMap = new Map<string, tf.Tensor>();
  }

  public override applyGradients(variableGradientsMap: any[]): void {
    tf.tidy(() => {
      // 1. Get the direct reference to the live engine variable registry
      const registeredVars = tf.engine().registeredVariables;

      // 2. Loop through the clean array passed by model.fit()
      for (const item of variableGradientsMap) {
        const varName = item.name;
        const grad = item.tensor; // The 'tensor' property is the computed gradient!

        // 3. Pull the actual mutable variable directly out of the registry by its string name
        const variable = registeredVars[varName] as tf.Variable;

        if (!variable || grad == null) continue;

        // 4. Standard basic SGD math: weight = weight - (learningRate * gradient)
        const delta = grad.mul(this.learningRate);
        const newValue = variable.sub(delta);

        // 5. Update the live weights instantly
        variable.assign(newValue);
      }
    });
  }

  private getOrCreateState(variable: tf.Variable, stateName: string): tf.Tensor {
    const key = `${variable.id}/${stateName}`;
    if (!this.stateMap.has(key)) {
      const initialState = tf.keep(tf.zerosLike(variable));
      this.stateMap.set(key, initialState);
    }
    return this.stateMap.get(key)!;
  }

  private updateStateTensor(variable: tf.Variable, stateName: string, newTensor: tf.Tensor): void {
    const key = `${variable.id}/${stateName}`;
    const oldTensor = this.stateMap.get(key);
    this.stateMap.set(key, tf.keep(newTensor));
    if (oldTensor) oldTensor.dispose();
  }

  public override dispose(): void {
    super.dispose();
    this.stateMap.forEach((tensor) => tensor.dispose());
    this.stateMap.clear();
  }

  public override getConfig(): tf.serialization.ConfigDict {
    return { learningRate: this.learningRate };
  }

  public static override fromConfig<T extends tf.serialization.Serializable>(cls: tf.serialization.SerializableConstructor<T>, config: tf.serialization.ConfigDict): T {
    const lr = (config as any).learningRate ?? 0.01;
    return new cls(lr);
  }

  public override getClassName(): string {
    return CustomBasicOptimizer.className;
  }
}
tf.serialization.SerializationMap.register(CustomBasicOptimizer);

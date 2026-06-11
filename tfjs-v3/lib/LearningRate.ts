export interface ILearningRate {
  getLearningRate(): number;
  incrementStep(): void;
}

export class LearningRate {
  private currentStep: number = 1;
  private schedulerFunc: (step: number) => number;

  /**
   * @param schedulerFunc A function that takes the step integer and returns a float learning rate.
   */
  constructor(schedulerFunc: (step: number) => number) {
    this.schedulerFunc = schedulerFunc;
  }

  // Sampled dynamically inside the layer's gradFunc
  public getLearningRate(): number {
    return this.schedulerFunc(this.currentStep);
  }

  // Triggered exactly once per batch by your custom optimizer
  public incrementStep(): void {
    this.currentStep++;
  }
}

import * as tf from "@tensorflow/tfjs";
import { expect } from "chai";
import { SparseMiniBitNetStrategy } from "../lib/strategy/SparseMiniBitNetStrategy";

describe.only("SparseDiscreteOptimizer Unit Tests", () => {
  let optimizer: SparseMiniBitNetStrategy;

  beforeEach(() => {
    optimizer = new SparseMiniBitNetStrategy();
  });

  describe("getApproximateZScore() Precision", () => {
    it("should calculate highly accurate Z-scores for standard industry learning rates", () => {
      // Test case: 5% target density -> Two-tailed Z-score should map exactly to 1.9600
      const zFor05 = optimizer.getApproximateZScore(0.05);
      expect(zFor05).to.be.closeTo(1.96, 0.001);

      // Test case: 1% target density -> Two-tailed Z-score should map exactly to 2.5758
      const zFor01 = optimizer.getApproximateZScore(0.01);
      expect(zFor01).to.be.closeTo(2.5758, 0.001);

      // Test case: 10% target density -> Two-tailed Z-score should map exactly to 1.6449
      const zFor10 = optimizer.getApproximateZScore(0.1);
      expect(zFor10).to.be.closeTo(1.6449, 0.001);
    });
  });

  describe("Gradient Gating Functional Behavior", () => {
    it("should filter out precisely 95% of elements given a learning rate of 0.05", async () => {
      const shape = [100, 100, 10]; // 100,000 total elements distributed across channels
      const totalElements = 100000;
      const targetLearningRate = 0.05;

      const mockGradient = tf.randomNormal(shape, 0, 1);
      const gatedOutput = optimizer.optmizeLearningRate(mockGradient, targetLearningRate);

      // Filter and count everything that isn't hard-gated to exactly 0
      const nonZeroMask = tf.cast(tf.notEqual(gatedOutput, tf.scalar(0)), "float32");
      const totalPassedTensor = tf.sum(nonZeroMask);
      const totalPassed = (await totalPassedTensor.data())[0];

      const observedDensity = totalPassed / totalElements;

      // Clean up WebGL memory handles immediately
      tf.dispose([mockGradient, gatedOutput, nonZeroMask, totalPassedTensor]);

      // Allow a tight 0.5% margin due to sampling variability of tf.randomNormal
      expect(observedDensity).to.be.closeTo(targetLearningRate, 0.005);
    });
  });
});

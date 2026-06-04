import * as assert from "assert";
import * as tf from "@tensorflow/tfjs-node";
import { BitNetLayer } from "../lib/BitNetLayer";
import { SgdMomentumStrategy } from "../lib/strategies/SgdMomentumStrategy";

describe("BitNetLayer Integration Unit Tests", () => {
  it("should assemble successfully and compute matching forward output shape dimensions", () => {
    tf.tidy(() => {
      const mockStrategy = new SgdMomentumStrategy(0.9);
      const initialWeightsMatrix = tf.zeros([32, 784]) as tf.Tensor2D;

      const layer = new BitNetLayer({
        units: 32,
        inputShape: [784],
        strategy: mockStrategy,
        weights: [initialWeightsMatrix],
      });

      // Assert shape allocations computed internally by the layer config handlers
      // Passing an explicitly nested array structure [Batch size placeholder, input channels]
      const outShape = layer.computeOutputShape([null, 784]);
      assert.deepStrictEqual(outShape, [null, 32]);
    });
  });

  it("should evaluate full-precision calculations and output valid float32 execution matrices", () => {
    tf.tidy(() => {
      const mockStrategy = new SgdMomentumStrategy(0.9);

      // Initialize an explicit 2x4 weights layer layout manually
      // Setup a mixture of negative and positive activations configurations
      const initialWeightsMatrix = tf.tensor2d(
        [
          [1.0, 0.0, -1.0, 0.0],
          [0.0, 1.0, 0.0, 1.0],
        ],
        [2, 4],
      );

      const layer = new BitNetLayer({
        units: 2,
        inputShape: [4],
        strategy: mockStrategy,
        weights: [initialWeightsMatrix],
      });

      // Pass matching multi-sample tensor data
      const inputActivations = tf.tensor2d(
        [
          [2.0, 3.0, 1.0, 5.0],
          [0.0, 1.0, 4.0, 2.0],
        ],
        [2, 4],
      );

      // Trigger the forward execution layer flow graph natively
      const outputTensor = layer.apply(inputActivations) as tf.Tensor;
      const resultsArray = outputTensor.dataSync();

      assert.deepStrictEqual(outputTensor.shape, [2, 2], "Output should match [Batch, Units]");
      assert.strictEqual(outputTensor.dtype, "float32", "Calculations must evaluate inside standard float containers");

      // Verification check: Batch element 0, unit 0: (2*1) + (3*0) + (1*-1) + (5*0) = 1
      assert.strictEqual(resultsArray[0], 1);
    });
  });
});

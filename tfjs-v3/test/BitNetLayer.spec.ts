import * as assert from "assert";
import * as tf from "@tensorflow/tfjs-node";
import { StrategyBitNetLayer } from "../lib/StrategyBitNetLayer";
import { SgdMomentumStrategy } from "../lib/strategies/SgdMomentumStrategy";

describe("BitNetLayer Integration Unit Tests", () => {
  it("should assemble successfully and compute matching forward output shape dimensions", () => {
    tf.tidy(() => {
      const mockStrategy = new SgdMomentumStrategy(0.9);
      const initialWeightsMatrix = tf.zeros([32, 784]) as tf.Tensor2D;

      const layer = new StrategyBitNetLayer({
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

      const layer = new StrategyBitNetLayer({
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

  it("should serialize its configuration properties correctly for framework saving", () => {
    const mockStrategy = new SgdMomentumStrategy(0.9);
    const layer = new StrategyBitNetLayer({
      units: 64,
      strategy: mockStrategy,
    });

    const config = layer.getConfig();

    assert.strictEqual(config.units, 64, "Serialized configuration dict must retain the units parameter value");
    assert.deepStrictEqual(config.strategy, mockStrategy, "Serialized configuration dict must preserve the strategy instance reference");
  });

  it("should correctly infer and extract input features from multi-dimensional shape arrays", () => {
    tf.tidy(() => {
      const mockStrategy = new SgdMomentumStrategy(0.9);
      const layer = new StrategyBitNetLayer({
        units: 10,
        strategy: mockStrategy,
      });

      // Pass a 4D tensor shape layout: [Batch size, Height, Width, Channels]
      // The build framework should evaluate the last index (128) as its inFeatures dimension width
      layer.build([null, 28, 28, 128]);

      // getPackedShape(10, 128) should allocate: [10, 128 / 4] = [10, 32]
      const kernelShape = layer.getWeights()[0].shape;
      assert.deepStrictEqual(kernelShape, [10, 32], "Layer must properly extract trailing channels from high-rank tensors");
    });
  });
});

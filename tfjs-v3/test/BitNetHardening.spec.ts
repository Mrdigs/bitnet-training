import * as assert from "assert";
import * as tf from "@tensorflow/tfjs-node";
import { SgdMomentumStrategy } from "../lib/strategies/SgdMomentumStrategy";
import { StrategyBitNetOptimizer } from "../lib/StrategyBitNetOptimizer";
import { StrategyBitNetLayer } from "../lib/StrategyBitNetLayer";

describe("BitNet Architecture Hardening & Stress Tests", () => {
  let strategy: SgdMomentumStrategy;
  const units = 16;
  const inFeatures = 64; // Must be divisible by 4

  beforeEach(() => {
    strategy = new SgdMomentumStrategy(0.9);
  });

  // =========================================================================
  // 1. BITFIELD BOUNDARY PROTECTION & OVERFLOW FUZZING
  // =========================================================================
  describe("Bitfield Boundary & State Isolation", () => {
    it("should guarantee that extreme continuous gradients do not corrupt adjacent bit fields", () => {
      const strategyPrivate = strategy as any;

      // Upper 2 bits: 1 (Weight token 1), Lower 6 bits: 31 (Max positive signed velocity)
      // Binary representation: 01 011111 = 95
      const initialByte = (1 << 6) | 31;

      // Fuzz with a massive negative gradient pushing velocity aggressively upward
      // continuousVelocity = (0.9 * 31) - (-1000.0 * 0.1 * 10) = 27.9 + 1000 = 1027.9
      // Rounded and clamped velocity must stop precisely at 31, while weight token shifts to 2 (mapped from 1 + 1)
      const resultByte = strategyPrivate.mutateElement(initialByte, -1000.0, 0.1);

      const updatedWeightToken = (resultByte >> 6) & 0x03;
      const updatedSixBitState = resultByte & 0x3f;

      // Assert that the overflow didn't break out of the 8-bit total element allocation byte boundaries
      assert.ok(resultByte <= 255, "Mutated byte must never exceed 8-bit integer boundaries.");
      assert.strictEqual(updatedWeightToken, 2, "Weight token must increment to 2 and not leak bits higher.");
      assert.strictEqual(updatedSixBitState, 0, "Velocity accumulator must flush back to zero upon crossing threshold.");
      assert.strictEqual(updatedWeightToken, 1, "Weight token must safely clamp at its maximum value of 1.");
    });

    it("should accurately maintain signed-to-unsigned two's complement boundary loops", () => {
      const strategyPrivate = strategy as any;

      // Start with negative maximum velocity boundary state: -32 (represented unsigned as 32)
      // Binary layout: 00 100000 = 32
      const initialByte = (0 << 6) | 32;

      // Push it further negative using a huge positive gradient force
      const resultByte = strategyPrivate.mutateElement(initialByte, 500.0, 0.1);

      const updatedWeightToken = (resultByte >> 6) & 0x03;
      const updatedSixBitState = resultByte & 0x3f;

      assert.strictEqual(updatedWeightToken, 2, "Weight token should wrap/shift down to -1 (unsigned token 2).");
      assert.strictEqual(updatedSixBitState, 0, "Velocity accumulator must clear back to zero on flip.");
    });
  });

  // =========================================================================
  // 2. UNMANAGED HEAP MEMORY TRANSITIONS & LEAK DETECTION
  // =========================================================================
  describe("Memory Safety & Tensor Cleanup", () => {
    it("should guarantee zero unmanaged memory leaks during intense dataSync iterations", () => {
      const initialMemory = tf.memory();

      // Instantiate our runtime components
      const optimizer = new StrategyBitNetOptimizer(strategy, 0.01);
      const packedContainer = tf.variable(tf.zeros([units, inFeatures / 4], "float32"));
      const gradientMock = tf.randomNormal([units, inFeatures]);

      // Execute a heavy training simulation batch loop block
      for (let step = 0; step < 20; step++) {
        tf.tidy(() => {
          const namedGradients: tf.NamedTensorMap = {
            [packedContainer.name]: gradientMock,
          };

          // Trigger the optimizer dataSync pipeline
          optimizer.applyGradients(namedGradients);
        });
      }

      const finalMemory = tf.memory();

      // Assert that the active tracking tensor allocations did not balloon on the native C++ heap
      assert.strictEqual(
        finalMemory.numTensors,
        initialMemory.numTensors + 2, // Only the newly allocated packedContainer and gradientMock should persist
        "Optimizer dataSync operations must fully collect all intermediate computational graph textures.",
      );

      // Clean up local scopes
      packedContainer.dispose();
      gradientMock.dispose();
    });
  });

  // =========================================================================
  // 3. ZERO-GRADIENT INVARIANCE GUARD
  // =========================================================================
  describe("Optimization Continuity & Edge Case Safety", () => {
    it("should keep weights and velocity states completely frozen when receiving a dead zero gradient", () => {
      const optimizer = new StrategyBitNetOptimizer(strategy, 0.05);

      // Establish an explicit baseline variable layout configuration
      const initialWeights = tf.randomUniform([units, inFeatures], -1, 1) as tf.Tensor2D;
      const packedWeights = strategy.prepareInitialWeights(initialWeights) as tf.Variable;

      const snapshotBefore = packedWeights.dataSync().slice();
      const zeroGradients = tf.zeros([units, inFeatures]);

      tf.tidy(() => {
        const namedGradients: tf.NamedTensorMap = {
          [packedWeights.name]: zeroGradients,
        };
        optimizer.applyGradients(namedGradients);
      });

      const snapshotAfter = packedWeights.dataSync();

      // Verify down to the exact float bitwise parity that absolutely zero drift happened
      assert.deepStrictEqual(Array.from(snapshotAfter), Array.from(snapshotBefore), "A zeroed-out gradient matrix must result in a perfect dead-pass with zero model mutations.");

      initialWeights.dispose();
      packedWeights.dispose();
      zeroGradients.dispose();
    });

    it("should correctly handle forward call operations under dynamic batch sizing variations", () => {
      tf.tidy(() => {
        const initialWeightsMatrix = tf.zeros([units, inFeatures]) as tf.Tensor2D;
        const layer = new StrategyBitNetLayer({
          units: units,
          strategy: strategy,
          weights: [initialWeightsMatrix],
        });

        // Test single standalone item batch size [1, inFeatures]
        const singleBatchInput = tf.randomUniform([1, inFeatures]);
        const singleOutput = layer.apply(singleBatchInput) as tf.Tensor;
        assert.deepStrictEqual(singleOutput.shape, [1, units]);

        // Test heavy deployment scaling batch size [128, inFeatures]
        const massiveBatchInput = tf.randomUniform([128, inFeatures]);
        const massiveOutput = layer.apply(massiveBatchInput) as tf.Tensor;
        assert.deepStrictEqual(massiveOutput.shape, [128, units]);
      });
    });
  });
});

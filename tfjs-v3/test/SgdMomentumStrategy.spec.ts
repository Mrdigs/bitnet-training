import * as assert from "assert";
import * as tf from "@tensorflow/tfjs-node";
import { SgdMomentumStrategy } from "../lib/strategies/SgdMomentumStrategy";

describe("SgdMomentumStrategy Bit-Packing Unit Tests", () => {
  let strategy: SgdMomentumStrategy;

  beforeEach(() => {
    strategy = new SgdMomentumStrategy(0.9); // Instantiate fresh strategy parameters
  });

  it("should calculate correct packed tracking shapes based on 4-way blocks", () => {
    // 784 inputs must collapse exactly to 196 elements per block row
    const packedShape = strategy.getPackedShape(64, 784);
    assert.deepStrictEqual(packedShape, [64, 196]);
  });

  it("should enforce divisible dimension checks on shape allocation", () => {
    assert.throws(() => {
      strategy.getPackedShape(64, 783); // 783 is not divisible by 4
    }, /must be a multiple of elementsPerBlock/);
  });

  it("should correctly serialize ternary weights to 2-bit storage tokens", () => {
    tf.tidy(() => {
      // Create explicit target baseline weights matrix: [-1.0, 0.0, 1.0, 0.0]
      const sourceFloats = tf.tensor2d([[-1.0, 0.0, 1.0, 0.0]], [1, 4]);

      const packedTensor = strategy.prepareInitialWeights(sourceFloats);

      // FIX: Extract the values as standard numbers from dataSync.
      // Since it is an integer value stored inside a float container, we round it safely.
      const packedData = packedTensor.dataSync();
      const word = Math.round(packedData[0]); // Extracted as the true literal integer: 4194432

      // Now your native bit-shifting masks operate perfectly on the actual value!
      const byte0 = word & 0xff; // Value: 128 (Token 2 shifted by 6)
      const byte1 = (word >> 8) & 0xff; // Value: 0
      const byte2 = (word >> 16) & 0xff; // Value: 64  (Token 1 shifted by 6)
      const byte3 = (word >> 24) & 0xff; // Value: 0

      assert.strictEqual(byte0 >> 6, 2, "Weight -1 must be mapped to token 2");
      assert.strictEqual(byte1 >> 6, 0, "Weight 0 must be mapped to token 0");
      assert.strictEqual(byte2 >> 6, 1, "Weight 1 must be mapped to token 1");
      assert.strictEqual(byte3 >> 6, 0, "Weight 0 must be mapped to token 0");
    });
  });

  it("should execute private element updates and clamp velocity arrays properly", () => {
    // Access and execute internal protected hook mutations using dynamic dictionary casting
    const strategyPrivateAccessor = strategy as any;

    // Test Case: Positive Gradient pushing positive velocity
    // Weight = 0, Unsigned State = 0 (Velocity 0). Learning rate = 0.1, Grad = 1.0
    // Continuous math: (0.9 * 0) - (1.0 * 0.1 * 10) = -1. Rounded: -1. Unsigned two's complement maps out to 63
    let result = strategyPrivateAccessor.mutateElement(0, 1.0, 0.1);

    let updatedWeightToken = (result >> 6) & 0x03;
    let updatedVelocityState = result & 0x3f;

    assert.strictEqual(updatedWeightToken, 0, "Weight shouldn't change yet on small force");
    assert.strictEqual(updatedVelocityState, 63, "Velocity should track signed value -1 as 63");

    // Test Case: Momentum trigger threshold flip boundary crossing execution
    // Weight = 0, Unsigned State = 15 (Velocity 15), strong push past threshold limit 16
    const highStateWord = (0 << 6) | 15;
    result = strategyPrivateAccessor.mutateElement(highStateWord, -1.0, 0.2); // Negative grad pushes velocity up

    updatedWeightToken = (result >> 6) & 0x03;
    updatedVelocityState = result & 0x3f;

    assert.strictEqual(updatedWeightToken, 1, "Weight should shift up by 1 because velocity cross threshold limits");
    assert.strictEqual(updatedVelocityState, 0, "Velocity accumulator field must flush to 0 after triggering updates");
  });
});

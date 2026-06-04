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

  // Inside test/SgdMomentumStrategy.spec.ts

  it("should execute private element updates and clamp velocity arrays properly", () => {
    const strategyPrivateAccessor = strategy as any;

    let result = strategyPrivateAccessor.mutateElement(0, 1.0, 0.1);
    let updatedWeightToken = (result >> 6) & 0x03;
    let updatedVelocityState = result & 0x3f;

    assert.strictEqual(updatedWeightToken, 0, "Weight shouldn't change yet on small force");
    assert.strictEqual(updatedVelocityState, 63, "Velocity should track signed value -1 as 63");

    // FIX: Set unsigned state to 17 (Velocity 17 / 8 = 2.125), which crosses the 2.0 flip threshold
    const highStateWord = (0 << 6) | 17;
    result = strategyPrivateAccessor.mutateElement(highStateWord, -1.0, 0.2);

    updatedWeightToken = (result >> 6) & 0x03;
    updatedVelocityState = result & 0x3f;

    assert.strictEqual(updatedWeightToken, 1, "Weight should shift up by 1 because velocity cross threshold limits");
    assert.strictEqual(updatedVelocityState, 0, "Velocity accumulator field must flush to 0 after triggering updates");
  });

  it("should gradually accumulate small fractional gradients over multiple updates", () => {
    const strategyPrivateAccessor = strategy as any;
    let element = (1 << 6) | 0; // Weight: 1, Velocity: 0

    // Small updates: Grad = -0.2, LR = 0.1. Force per step = -(-0.2 * 0.1) = +0.02
    for (let i = 0; i < 3; i++) {
      element = strategyPrivateAccessor.mutateElement(element, -0.2, 0.1);
    }

    const updatedWeightToken = (element >> 6) & 0x03;
    const updatedVelocityState = element & 0x3f;

    assert.strictEqual(updatedWeightToken, 1, "Weight must remain stable during minor accumulation phases");
    // Under Math.round, 3 steps of 0.02 accumulation * 8 resolution ticks register a clear positive value change!
    assert.ok(updatedVelocityState > 0, "Velocity tracking register must register incremental change accumulation");
  });

  it("should map tokens to precise ternary floats using polynomial math", () => {
    tf.tidy(() => {
      // Instantiate raw tokens: [0, 1, 2]
      const tokens = tf.tensor1d([0, 1, 2], "int32");

      // Access the protected mapping method via type-casting
      const strategyPrivate = strategy as any;
      const floatsTensor = strategyPrivate.mapTokensToTernaryFloats(tokens);
      const floats = Array.from(floatsTensor.dataSync());

      assert.strictEqual(floats[0], 0.0, "Token 0 must map precisely to 0.0");
      assert.strictEqual(floats[1], 1.0, "Token 1 must map precisely to 1.0");
      assert.strictEqual(floats[2], -1.0, "Token 2 must map precisely to -1.0");
    });
  });
});

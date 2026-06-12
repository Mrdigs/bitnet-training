import * as tf from "@tensorflow/tfjs-node";
import { expect } from "chai";
import { FourTo1BitPackingStrategy } from "../lib/strategy/FourTo1BitPackingStrategy";
import { PersistentState } from "../lib/PersistentState";

describe.only("FourTo1BitPackingStrategy Unit Tests", () => {
  let strategy: FourTo1BitPackingStrategy;
  let dummyState: PersistentState;

  beforeEach(() => {
    strategy = new FourTo1BitPackingStrategy();
    dummyState = new PersistentState();
  });

  afterEach(() => {
    tf.disposeVariables();
  });

  it("should calculate correct compressed structural shapes", () => {
    const packedShape = strategy.getPackedShape(10, 128);
    expect(packedShape).to.deep.equal([32, 10]);
  });

  it("should execute a lossless 4-to-1 initialization compression lifecycle", () => {
    const originalDense = tf.tensor2d(
      [
        [-1.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [-1.0, -1.0],
      ],
      [4, 2],
      "float32",
    );

    const packedStorage = strategy.prepareInitialWeights(originalDense);
    expect(packedStorage.dtype).to.equal("float32"); // Verifies the lie is successful

    const decodedDense = strategy.decodeWeights(packedStorage, dummyState);
    expect(Array.from(decodedDense.dataSync())).to.deep.equal(Array.from(originalDense.dataSync()));

    tf.dispose([originalDense, packedStorage, decodedDense]);
  });

  it("should execute a lossless roundtrip for both ternary weights and signed momentum registers at scale", () => {
    const targetShape: [number, number] = [128, 64];

    const randomWeightIndices = tf.randomUniform(targetShape, 0, 3, "int32");
    const randomMomentumValues = tf.randomUniform(targetShape, -32, 32, "int32");

    // Pack returns a spoofed float32
    const packedPayload = strategy.pack(randomWeightIndices, randomMomentumValues);
    expect(packedPayload.dtype).to.equal("float32");

    // Unpack automatically processes the conversion and decouples cleanly
    const { weight: decodedWeights, momentum: decodedMomentum } = strategy.unpack(packedPayload);

    const weightsAreLossless = tf.tidy(() => tf.all(tf.equal(randomWeightIndices, decodedWeights)).dataSync()[0] === 1);
    const momentumIsLossless = tf.tidy(() => tf.all(tf.equal(randomMomentumValues, decodedMomentum)).dataSync()[0] === 1);

    expect(weightsAreLossless).to.be.true;
    expect(momentumIsLossless).to.be.true;

    tf.dispose([randomWeightIndices, randomMomentumValues, packedPayload, decodedWeights, decodedMomentum]);
  });
});

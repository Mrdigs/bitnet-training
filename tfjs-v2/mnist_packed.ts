import * as tf from "@tensorflow/tfjs-node";
import { PackedBitNetLayer } from "./lib/PackedBitNetLayer";
import { PackedBitNetOptimizer } from "./lib/PackedBitNetOptimizer";
import { SimpleMomentumStrategy } from "./lib/strategy/SimpleMomentumStrategy";
import mnist from "mnist";

interface ProcessedTensors {
  xs: tf.Tensor2D;
  ys: tf.Tensor2D;
}

async function runTraining(): Promise<void> {
  console.log("Loading and preparing MNIST data...");

  // Consume the dataset directly from the asynchronously loaded module object
  const mnistData = mnist.set(10000, 2000);

  function convertToTensors(dataSubset: any[]): ProcessedTensors {
    const images: number[][] = [];
    const labels: number[][] = [];

    for (const sample of dataSubset) {
      images.push(sample.input);
      labels.push(sample.output);
    }

    const xs = tf.tensor2d(images, [images.length, 784]);
    const ys = tf.tensor2d(labels, [labels.length, 10]);
    return { xs, ys };
  }

  const trainData = convertToTensors(mnistData.training);
  const testData = convertToTensors(mnistData.test);

  console.log("Building the model architecture with bit-packed BitNet layers...");

  const strategy = new SimpleMomentumStrategy(0.9);
  const customOptimizer = new PackedBitNetOptimizer(strategy, 0.01);

  const model = tf.sequential();

  model.add(
    new PackedBitNetLayer({
      inputShape: [784],
      units: 128,
      strategy: strategy,
      activation: "relu",
    }),
  );

  model.add(
    new PackedBitNetLayer({
      units: 10,
      strategy: strategy,
      activation: "softmax",
    }),
  );

  model.compile({
    optimizer: customOptimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  console.log("Starting training process with bit-packed layers...");

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 5,
    batchSize: 64,
    validationData: [testData.xs, testData.ys],
    callbacks: {
      onEpochEnd: (epoch: number, logs?: tf.Logs) => {
        if (logs) {
          // Diagnostic: Extract a sample slice from the first layer to verify bit mutations
          const layer1 = model.layers[0] as PackedBitNetLayer;
          const sampleTensor = tf.cast(layer1.packedKernel.read(), "float32");
          const sampleData = sampleTensor.slice([0, 0], [1, 5]).dataSync(); // Read first 5 weights

          console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, ` + `Accuracy = ${logs.acc.toFixed(4)}, ` + `Val Accuracy = ${logs.val_acc.toFixed(4)} | ` + `Packed Weight Sample Context: [${Array.from(sampleData).join(", ")}]`);
        }
      },
    },
  });

  console.log("\nTraining complete!");
  console.log("Evaluating test dataset...");

  const evalResult = model.evaluate(testData.xs, testData.ys) as tf.Tensor[];

  const testLoss = evalResult[0].dataSync()[0];
  const testAcc = evalResult[1].dataSync()[0];

  console.log(`Final Test Loss: ${testLoss.toFixed(4)}`);
  console.log(`Final Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);

  trainData.xs.dispose();
  trainData.ys.dispose();
  testData.xs.dispose();
  testData.ys.dispose();
}

// Kick off the async runner wrapper loop
runTraining().catch((err) => console.error(err));

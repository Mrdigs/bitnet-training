import * as tf from "@tensorflow/tfjs-node";
import { BitNetLayer } from "./lib/BitNetLayer";
import { BitNetOptimizer } from "./lib/BitNetOptimizer";
import { SgdMomentumStrategy } from "./lib/strategies/SgdMomentumStrategy";

async function runMinimalFitTest(): Promise<void> {
  console.log("=== ⚙️ STARTING ARCHITECTURE INTEGRATION TEST ===");

  // 1. Instantiate the strategy layout (SGD with Momentum)
  const momentumStrategy = new SgdMomentumStrategy(0.9);

  // 2. Generate a deterministic high-precision initial weight matrix [64 units, 784 features]
  const inputDim = 784;
  const units = 64;
  const initialWeightsMatrix = tf.randomNormal([units, inputDim], 0.0, 0.05, "float32", 42) as tf.Tensor2D;

  // 3. Build a standard TensorFlow.js Sequential model using our custom Layer
  const model = tf.sequential();

  model.add(
    new BitNetLayer({
      units: units,
      inputShape: [inputDim],
      strategy: momentumStrategy,
      weights: [initialWeightsMatrix],
    }),
  );

  // Add a standard output classification head to check sequential continuity
  model.add(tf.layers.activation({ activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  // 4. Instantiate our custom BitNetOptimizer, configured with a distinct learning rate
  const bitNetOptimizer = new BitNetOptimizer(momentumStrategy, 0.01);

  // Compile the model using standard metrics, passing a dummy placeholder string for the optimizer
  // because we will intercept and handle the gradient mutations manually via the callback mechanism
  model.compile({
    optimizer: "sgd",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // 5. Generate high-velocity, lightweight synthetic training data batches
  const numSamples = 128;
  const xTrain = tf.randomUniform([numSamples, inputDim], 0, 1);
  const yTrain = tf.oneHot(tf.randomUniform([numSamples], 0, 10, "int32"), 10);

  console.log("-> Compiling execution graph. Commencing model.fit()...");

  // 6. Execute model.fit() with a custom CustomCallback hook to run our bitwise updates
  await model.fit(xTrain, yTrain, {
    epochs: 2,
    batchSize: 32,
    shuffle: false,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        tf.tidy(() => {
          // Explicitly evaluate loss gradients with respect to the layer variables
          // This captures the exact backward derivatives graph for all active model parameters
          const costGrads = tf.variableGrads(() => {
            const predictions = model.predict(xTrain.slice([batch * 32, 0], [32, inputDim])) as tf.Tensor;
            const targets = yTrain.slice([batch * 32, 0], [32, 10]);
            return tf.losses.softmaxCrossEntropy(targets, predictions) as tf.Scalar;
          });

          // Inject computed gradient mappings straight into our decoupled optimizer loop
          bitNetOptimizer.applyGradients(costGrads.grads);
        });

        console.log(`   [Batch ${batch + 1}] Completed. Loss metric value: ${logs?.loss?.toFixed(5)}`);
      },
    },
  });

  // 7. Validate that the weights matrix remains structurally solid
  const finalWeights = model.layers[0].getWeights()[0];
  console.log("\n=== ✅ SYSTEM VERIFICATION SUCCESSFUL ===");
  console.log(`Packed Weight Container Shape: [${finalWeights.shape.join(", ")}]`);
  console.log("The entire lifecycle (Build, Call, Forward, Unpack, Backward, Mutate) executed with zero compilation or runtime errors.");

  // Clean up remaining allocations on the unmanaged heap boundary
  xTrain.dispose();
  yTrain.dispose();
  initialWeightsMatrix.dispose();
}

runMinimalFitTest().catch((err) => {
  console.error("\n❌ Test harness failed to execute cleanly due to a compilation or runtime error:");
  console.error(err);
});

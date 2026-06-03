import * as tf from "@tensorflow/tfjs";
import mnist from "mnist";

// ===================================================
// 1. Define the BitNet 1.58b Layer (Quantized + STE)
// ===================================================
class BitNet158Layer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.units = config.units;
    this.activationName = config.activation || "linear";

    if (this.activationName === "relu") {
      this.activation = tf.relu;
    } else if (this.activationName === "softmax") {
      this.activation = tf.softmax;
    } else {
      this.activation = (x) => x;
    }
  }

  build(inputShape) {
    const inputDim = inputShape[inputShape.length - 1];

    // Maintain full-precision latent ("shadow") weights for optimizer updates
    this.kernel = this.addWeight("kernel", [inputDim, this.units], "float32", tf.initializers.glorotUniform());

    // Keep bias at standard FP32 precision (standard practice in BitNet)
    this.bias = this.addWeight("bias", [this.units], "float32", tf.initializers.zeros());

    this.built = true;
  }

  computeOutputShape(inputShape) {
    const outputShape = [...inputShape];
    outputShape[outputShape.length - 1] = this.units;
    return outputShape;
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      // In tfjs layers, inputs can be an array containing a single tensor
      const inputTensor = Array.isArray(inputs) ? inputs[0] : inputs;
      const fullPrecisionWeights = this.kernel.read();

      // --- Straight-Through Estimator (STE) via tf.customGrad ---
      // tf.customGrad takes a function that returns an object: { value: tensor, gradFunc: function }
      const steQuantize = tf.customGrad((fullWeights) => {
        // --- BitNet 1.58b Ternary Quantization Pipeline ---
        // 1. Calculate the per-tensor scaling factor (AbsMean)
        const scale = tf.mean(tf.abs(fullWeights));

        // 2. Prevent division-by-zero errors
        const safeScale = tf.add(scale, 1e-7);

        // 3. Quantize full precision weights down to {-1, 0, 1}
        const quantizedWeights = tf.clipByValue(tf.round(tf.div(fullWeights, safeScale)), -1, 1);

        // 4. Scale back up to continuous space (Fake Quantization)
        const dequantizedWeights = tf.mul(quantizedWeights, safeScale);

        // Define the backward pass strategy
        // dy is the upstream gradient.
        // We must return an array of gradients matching the inner function parameters.
        const gradFunc = (dy) => {
          return [dy]; // Directly forwards upstream gradient to fullWeights
        };

        return { value: dequantizedWeights, gradFunc };
      });

      // Compute quantized weights using our custom STE function
      const steWeights = steQuantize(fullPrecisionWeights);

      // --- Compute Matrix Multiplication ---
      const linearOutput = tf.add(tf.matMul(inputTensor, steWeights), this.bias.read());

      return this.activation(linearOutput);
    });
  }

  static get className() {
    return "BitNet158Layer";
  }
}

// Register the class so TensorFlow.js can recognize it
tf.serialization.registerClass(BitNet158Layer);

// ==========================================
// 2. Prepare the MNIST Data
// ==========================================
console.log("Loading and preparing MNIST data...");
const mnistData = mnist.set(10000, 2000); // Scaled down sample sizing for speed testing

function convertToTensors(dataSubset) {
  const images = [];
  const labels = [];
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

// ==========================================
// 3. Build Model Using BitNet 1.58b Layers
// ==========================================
console.log("Building the model architecture with BitNet 1.58b layers...");
const model = tf.sequential();

model.add(
  new BitNet158Layer({
    inputShape: [784],
    units: 128,
    activation: "relu",
  }),
);

model.add(
  new BitNet158Layer({
    units: 10,
    activation: "softmax",
  }),
);

// ==========================================
// 4. Compile and Train Model
// ==========================================
model.compile({
  optimizer: tf.train.adam(0.01), // Slightly higher learning rate often helps quantized networks adapt
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

async function runTraining() {
  console.log("Starting training process with BitNet layers...");

  await model.fit(trainData.xs, trainData.ys, {
    epochs: 5,
    batchSize: 64,
    validationData: [testData.xs, testData.ys],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}, Accuracy = ${logs.acc.toFixed(4)}, Val Accuracy = ${logs.val_acc.toFixed(4)}`);
      },
    },
  });

  console.log("\nTraining complete!");

  console.log("Evaluating test dataset...");
  const evalResult = model.evaluate(testData.xs, testData.ys);
  const testLoss = evalResult[0].dataSync()[0];
  const testAcc = evalResult[1].dataSync()[0];

  console.log(`Final Test Loss: ${testLoss.toFixed(4)}`);
  console.log(`Final Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);

  // Clean up tensors from memory
  trainData.xs.dispose();
  trainData.ys.dispose();
  testData.xs.dispose();
  testData.ys.dispose();
}

runTraining();

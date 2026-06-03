import * as tf from "@tensorflow/tfjs";
import mnist from "mnist";

// ==========================================
// 1. Define the Custom Layer Implementation
// ==========================================
class CustomDenseLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.units = config.units;
    this.activationName = config.activation || "linear";

    // Resolve activation function string to an actual TF function
    if (this.activationName === "relu") {
      this.activation = tf.relu;
    } else if (this.activationName === "softmax") {
      this.activation = tf.softmax;
    } else {
      this.activation = (x) => x; // Linear (pass-through)
    }
  }

  // Allocate trainable weights (Weights and Biases) based on input dimensions
  build(inputShape) {
    // inputShape is an array, e.g., [batchSize, inputDim]
    const inputDim = inputShape[inputShape.length - 1];

    // Create Weight Matrix (Kernel)
    this.kernel = this.addWeight(
      "kernel",
      [inputDim, this.units],
      "float32",
      tf.initializers.glorotUniform(), // Standard default initializer
    );

    // Create Bias Vector
    this.bias = this.addWeight(
      "bias",
      [this.units],
      "float32",
      tf.initializers.zeros(), // Standard default initializer
    );

    this.built = true;
  }

  // Compute the shape transformation
  computeOutputShape(inputShape) {
    // Replaces the last dimension (inputDim) with output units
    const outputShape = [...inputShape];
    outputShape[outputShape.length - 1] = this.units;
    return outputShape;
  }

  // Perform forward pass math: Output = Activation( (Input * Kernel) + Bias )
  call(inputs, kwargs) {
    return tf.tidy(() => {
      // inputs can be a single Tensor or an array of Tensors
      const inputTensor = Array.isArray(inputs) ? inputs[0] : inputs;

      // Linear transformation: (X * W) + b
      const linearOutput = tf.add(tf.matMul(inputTensor, this.kernel.read()), this.bias.read());

      // Apply activation function
      return this.activation(linearOutput);
    });
  }

  // Class configuration metadata required for serialization mechanics
  static get className() {
    return "CustomDenseLayer";
  }
}

// Register custom class with the TF framework to make it fully operational
tf.serialization.registerClass(CustomDenseLayer);

// ==========================================
// 2. Prepare the MNIST Data
// ==========================================
console.log("Loading and preparing MNIST data...");
const mnistData = mnist.set(60000, 10000);

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
// 3. Build Model Using the Custom Layers
// ==========================================
console.log("Building the model architecture with custom layers...");
const model = tf.sequential();

// Hidden custom layer: 784 inputs -> 128 hidden units with ReLU
model.add(
  new CustomDenseLayer({
    inputShape: [784],
    units: 128,
    activation: "relu",
  }),
);

// Output custom layer: 128 inputs -> 10 units with Softmax
model.add(
  new CustomDenseLayer({
    units: 10,
    activation: "softmax",
  }),
);

// ==========================================
// 4. Compile and Train Model
// ==========================================
model.compile({
  optimizer: tf.train.adam(),
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

async function runTraining() {
  console.log("Starting training process with custom layers...");

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

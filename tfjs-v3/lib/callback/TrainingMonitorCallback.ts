import * as tf from "@tensorflow/tfjs-node";
import process from "node:process";

export class TrainingMonitorCallback extends tf.CustomCallback {
  // Memory snapshots taken before training starts
  private startTensorBytes: number = 0;
  private startProcessRss: number = 0;

  constructor() {
    super({
      // Hook 2: Fires at the end of every individual epoch
      onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
        if (logs) {
          const loss = logs.loss ?? 0;
          const acc = logs.acc ?? logs.accuracy ?? 0;
          const valAcc = logs.val_acc ?? logs.val_accuracy ?? 0;

          console.log(`Epoch ${epoch + 1}: Loss = ${loss.toFixed(4)}, Accuracy = ${acc.toFixed(4)}, Val Accuracy = ${valAcc.toFixed(4)}`);
        }
      },

      // Hook 3: Fires once training is completely finished
      onTrainEnd: async (logs) => {
        const endTensorMem = tf.memory();
        const endProcessMem = process.memoryUsage();

        // Calculate net overhead by subtracting data baselines
        const netTensorBytes = endTensorMem.numBytes - this.startTensorBytes;
        const netProcessRss = endProcessMem.rss - this.startProcessRss;

        // Conversion to Megabytes
        const netTensorMB = (netTensorBytes / (1024 * 1024)).toFixed(2);
        const netProcessMB = (netProcessRss / (1024 * 1024)).toFixed(2);

        console.log(`\n====================================`);
        console.log(`=== FINAL TRAINING MEMORY REPORT ===`);
        console.log(`====================================`);
        console.log(`* Total End Tensors   : ${endTensorMem.numTensors}`);
        console.log(`* Net Tensor Overhead : ${netTensorMB} MB (Excluding Inputs)`);
        console.log(`* Net System RSS Growth: ${netProcessMB} MB`);
        console.log(`====================================\n`);
      },
    });
    this.startTensorBytes = tf.memory().numBytes;
    this.startProcessRss = process.memoryUsage().rss;
  }
}

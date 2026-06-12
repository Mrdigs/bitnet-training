import * as tf from "@tensorflow/tfjs-node";
import * as asciichart from "asciichart";

export class TerminalChartCallback extends tf.CustomCallback {
  private lossHistory: number[] = [];
  private chartConfig: any;
  private readonly targetWidth: number;

  constructor(height = 12, maxWidth = 90) {
    super({
      // Collect metrics silently at the end of each training batch
      onBatchEnd: async (batch: number, logs?: tf.Logs) => {
        if (logs && typeof logs.loss === "number") {
          this.lossHistory.push(logs.loss);
        }
      },
      // Print the compact width-restricted chart when training completely finishes
      onTrainEnd: async (logs?: tf.Logs) => {
        console.log(`\n  📈 TRAINING LOSS HISTORY CHART (Max Width: ${this.targetWidth} Chars):`);
        console.log(`  ================================================================================`);

        if (this.lossHistory.length >= 2) {
          let displayData = this.lossHistory;

          // Downsample the metrics matrix array if it exceeds our horizontal space limits
          if (this.lossHistory.length > this.targetWidth) {
            displayData = [];
            const stride = this.lossHistory.length / this.targetWidth;
            for (let i = 0; i < this.targetWidth; i++) {
              const targetIndex = Math.floor(i * stride);
              displayData.push(this.lossHistory[targetIndex]);
            }
          }

          console.log(asciichart.plot(displayData, this.chartConfig));
        } else {
          console.log("  [ Not enough metrics gathered to generate a line graph profile ]");
        }

        console.log(`  ================================================================================\n`);
      },
    });

    this.targetWidth = maxWidth;
    this.chartConfig = {
      height: height,
      colors: [asciichart.green],
    };
  }
}

import * as tf from "@tensorflow/tfjs-node";

export class FirstPassLossCallback extends tf.CustomCallback {
  constructor() {
    super({
      // Hook fires instantly when the very first batch completes execution
      onBatchEnd: async (batch: number, logs?: tf.Logs) => {
        if (batch === 0) {
          console.log(`\n  🚨 UNMUTATED INITIAL STEP 0 EVALUATION:`);
          console.log(`  =======================================`);
          console.log(`  True First Forward Pass Loss: ${logs?.loss?.toFixed(4)}`);
          console.log(`  True First Forward Pass Acc:  ${logs?.acc?.toFixed(4)}`);
          console.log(`  =======================================\n`);
        }
      },
    });
  }
}

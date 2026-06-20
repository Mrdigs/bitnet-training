import * as tf from "@tensorflow/tfjs";

export class TernaryWeightDistributionCallback extends tf.CustomCallback {
  private layerName: string;
  private targetModel: tf.LayersModel;

  /**
   * @param model The active tf.LayersModel instance being trained.
   * @param layerName The name or keyword substring of your StrategyBitNetLayer instance.
   */
  constructor(model: tf.LayersModel, layerName: string = "strategy_bit_net") {
    // Pass empty configurations down to satisfy base architecture requirements
    super({});
    this.targetModel = model;
    this.layerName = layerName;
  }

  public override async onEpochEnd(epoch: number, logs?: tf.Logs): Promise<void> {
    // 1. Safeguard check on the manually passed model instance
    if (!this.targetModel || !this.targetModel.layers) {
      console.warn(`\n⚠️ [Distribution Callback]: Provided model instance is invalid.`);
      return;
    }

    // 2. Locate the custom layer within the underlying model topology
    const targetLayer = this.targetModel.layers.find((l: any) => l.name.includes(this.layerName));

    if (!targetLayer) {
      console.warn(`\n⚠️ [Distribution Callback]: Could not find a layer matching name "${this.layerName}"`);
      return;
    }

    // 3. Wrap memory evaluation inside a tidy block to prevent VRAM accumulation
    const stats = tf.tidy(() => {
      // Fetch the raw GPU variable holding the packed parameters
      const packedWeights = (targetLayer as any).kernelVar.read();
      const state = (targetLayer as any).layerState;

      // Decode the compressed bitmask back into the executed ternary matrix (-1.0, 0.0, 1.0)
      const decodedTernary = (targetLayer as any).strategy.decodeWeights(packedWeights, state);

      // Calculate total parameter count
      const totalElements = decodedTernary.size;

      // Establish masks to count the structural allocation boundaries
      const negOneMask = tf.less(decodedTernary, tf.scalar(-0.5));
      const posOneMask = tf.greater(decodedTernary, tf.scalar(0.5));

      // Deduce the middle neutral states by subtracting outer states from total boundaries
      const countNegOnes = tf.sum(negOneMask.toInt());
      const countPosOnes = tf.sum(posOneMask.toInt());
      const countZeros = tf.sub(tf.scalar(totalElements), tf.add(countNegOnes, countPosOnes));

      // Calculate percentages entirely on the GPU to maximize performance
      const pctNeg = tf.div(tf.mul(countNegOnes, tf.scalar(100)), tf.scalar(totalElements));
      const pctZero = tf.div(tf.mul(countZeros, tf.scalar(100)), tf.scalar(totalElements));
      const pctPos = tf.div(tf.mul(countPosOnes, tf.scalar(100)), tf.scalar(totalElements));

      // Keep these tracking tensors alive to read their values outside the tidy block
      return {
        total: totalElements,
        neg: pctNeg,
        zero: pctZero,
        pos: pctPos,
      };
    });

    // 4. Extract the scalar numbers out of WebGL/WASM memory synchronously
    const pNeg = (await stats.neg.data())[0];
    const pZero = (await stats.zero.data())[0];
    const pPos = (await stats.pos.data())[0];

    // 5. Instantly clean up the tracking metrics to keep memory footprint flat
    stats.neg.dispose();
    stats.zero.dispose();
    stats.pos.dispose();

    // 6. Print a clean, scannable dashboard directly to the console output
    console.log(`\n==================================================`);
    console.log(`📊 [Epoch ${epoch + 1}] Ternary Structural Allocation Status:`);
    console.log(`   Total Network Weights Checked : ${stats.total.toLocaleString()}`);
    console.log(`   -----------------------------------------------`);
    console.log(`   🔴 Negative States [-1.0]      : ${pNeg.toFixed(2)}%`);
    console.log(`   ⚪ Neutral States  [ 0.0]      : ${pZero.toFixed(2)}%  ${pZero < 5.0 ? "⚠️ (Potential Bridge Depletion)" : "✅"}`);
    console.log(`   🔵 Positive States [+1.0]      : ${pPos.toFixed(2)}%`);
    console.log(`==================================================\n`);
  }
}

import * as tf from "@tensorflow/tfjs";
import * as chart from "asciichart";

const STEPS = 150;
const GRAD_SCALE = 15.0;
const LAMBDA = 0.82;
const CRADLE_THRESHOLD = 0.05; // Lock parameters into rest if total input stays under noise floor

// Right-Side-Up Threshold Accumulator LUT Data Map
const LUT_DATA = [0.0, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.044, 0.055, 0.069, 0.086, 0.107, 0.134, 0.168, 0.21, 0.262, 0.328, 0.41, 0.512, 0.64, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

function runNoisyFusedWeightSimulation() {
  console.log("=========================================================");
  console.log(`LAUNCHING NOISE-POLLUTED STRESS TEST: ${STEPS} STEPS       `);
  console.log("=========================================================");

  const gradientHistory: number[] = [];
  const momentumHistory: number[] = [];
  const weightHistory: number[] = [];

  // Continuous floating-point tracking registers
  let m = 0.0;
  let w = 1; // Unsigned token value mapping (0=-1, 1=0, 2=+1)

  console.log("Injecting high-frequency background noise. Computing loops...\n");

  for (let t = 0; t < STEPS; t++) {
    let structuralGrad = 0.0;

    // Define the baseline underlying target gradient trajectory
    if (t >= 0 && t < 35) structuralGrad = -0.25;
    if (t >= 35 && t < 85) structuralGrad = 0.0;
    if (t >= 85 && t < 125) structuralGrad = 0.25;
    if (t >= 125 && t < 150) structuralGrad = 0.0;

    // NOISE INJECTION MATRIX: Generate continuous uniform jitter bounded between -0.05 and +0.05
    const noiseJitter = (Math.random() - 0.5) * 0.1;

    // The actual gradient context experienced by the layer is completely polluted with static noise
    const dynamicGradInput = structuralGrad + noiseJitter;
    gradientHistory.push(dynamicGradInput);

    const gradMag = Math.abs(dynamicGradInput);
    const gradSign = -Math.sign(dynamicGradInput);

    // --- BRANCHLESS TWO-LINE CORE WITH DYNAMIC STEP SIZE ---
    if (gradMag >= CRADLE_THRESHOLD && dynamicGradInput !== 0) {
      const prob = Math.min(1.0, gradMag * GRAD_SCALE);
      if (Math.random() < prob) {
        // Step size adjusts dynamically to the noisy input magnitude
        const dynamicStepSize = Math.max(1.0, Math.round(gradMag * 24.0));

        if (Math.sign(m) === gradSign || m === 0) m += gradSign * dynamicStepSize;
        else m -= Math.sign(m) * 6.0; // Enforce our linear braking filter
      }
      m = m * LAMBDA; // Apply un-gated time decay friction
    } else {
      m = 0.0; // Hard cradle lock handles the background static noise natively!
    }
    m = Math.max(-32.0, Math.min(31.0, m));

    // Stochastic Rounding Layer 1 Gate
    const floorMom = Math.floor(m);
    const fractionalPart = m - floorMom;
    const currentMomentumInt = Math.random() < fractionalPart ? floorMom + 1 : floorMom;

    // --- STOCHASTIC GATE STRATEGY WITH ZERO-CROSSING FILTER ---
    const scale_t = 1.0 + 0.5 * (1.0 - t / STEPS);
    const absVelocity = Math.abs(currentMomentumInt);
    const annealedIndex = Math.max(0, Math.min(32, Math.floor(absVelocity * scale_t)));
    const flipProbability = LUT_DATA[annealedIndex];

    if (Math.random() < flipProbability) {
      const isMomPositive = currentMomentumInt > 0;
      const isMomNegative = currentMomentumInt < 0;

      if (isMomPositive && w < 2) {
        w += 1; // Step up by 1 token gracefully
        m = m * 0.5; // Wash-out velocity damping on state change
      } else if (isMomNegative && w > 0) {
        w -= 1; // Step down by 1 token gracefully
        m = m * 0.5;
      }
    }

    const realWeightFloat = w - 1;
    momentumHistory.push(currentMomentumInt);
    weightHistory.push(realWeightFloat);
  }

  // --- AUTOMATED CHARACTER-PADDED LAYOUT RENDERER ---
  const plotAndAlignAxis = (history: number[], height: number, color: string) => {
    const rawChartStr = chart.plot(history, { height, colors: [color] } as any);
    const chartLines = rawChartStr.split("\n");

    let maxYLabelLength = 0;
    for (let i = 0; i < chartLines.length; i++) {
      const match = chartLines[i].match(/^\s*[\d\.\-]+/);
      if (match && match.length > maxYLabelLength) {
        maxYLabelLength = match.length;
      }
    }

    const pad = " ".repeat(maxYLabelLength + 2);
    const axisLine = pad + "└" + "─".repeat(history.length - 1);

    let labelRow = pad + "0";
    labelRow += " ".repeat(31) + "35";
    labelRow += " ".repeat(46) + "85";
    labelRow += " ".repeat(36) + "125";
    labelRow += " ".repeat(16) + "150 (Steps)";

    console.log(rawChartStr);
    console.log(axisLine);
    console.log(labelRow + "\n");
  };

  console.log("\n#########################################################");
  console.log("VISUAL TRANSITION REPORT: NOISE RESILIENCE DATA MAP      ");
  console.log("#########################################################");

  console.log("[1. NOISY INCOMING GRADIENTS JITTER]");
  plotAndAlignAxis(gradientHistory, 6, chart.red);

  console.log("[2. RESULTING INTERNAL REGISTER SIGNED MOMENTUM]");
  plotAndAlignAxis(momentumHistory, 8, chart.cyan);

  console.log("[3. ACTIVE TERNARY WEIGHT POSITION WITH LOCK]");
  plotAndAlignAxis(weightHistory, 6, chart.green);
  console.log("=========================================================");
}

runNoisyFusedWeightSimulation();

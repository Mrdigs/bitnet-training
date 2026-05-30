import * as tf from "@tensorflow/tfjs";
import * as chart from "asciichart";

const STEPS = 150; // Scaled runway as requested
const GRAD_SCALE = 15.0;
const LAMBDA = 0.82;

// Right-Side-Up Threshold Accumulator LUT Data Map
const LUT_DATA = [0.0, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.044, 0.055, 0.069, 0.086, 0.107, 0.134, 0.168, 0.21, 0.262, 0.328, 0.41, 0.512, 0.64, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

function runFusedWeightSimulation() {
  console.log("=========================================================");
  console.log(`LAUNCHING RUNWAY BENCHMARK RUN: ${STEPS} STEP ARENA     `);
  console.log("=========================================================");

  const gradientHistory: number[] = [];
  const momentumHistory: number[] = [];
  const weightHistory: number[] = [];

  // Continuous floating-point tracking registers
  let m = 0.0;
  let w = 1; // Unsigned token token value mapping (0=-1, 1=0, 2=+1)

  console.log("Processing continuous data manifolds. Computing loops...");

  for (let t = 0; t < STEPS; t++) {
    let currentGradValue = 0.0;

    // Define the scaled 150-step physical stress trajectory
    if (t >= 0 && t < 35) currentGradValue = -0.25; // Phase 1: Build positive momentum (+1 target)
    if (t >= 35 && t < 85) currentGradValue = 0.0; // Phase 2: Steady valley cradle rest hold
    if (t >= 85 && t < 125) currentGradValue = 0.25; // Phase 3: Build negative momentum (-1 target)
    if (t >= 125 && t < 150) currentGradValue = 0.0; // Phase 4: Final rest stabilization

    gradientHistory.push(currentGradValue);

    // --- THE TWO-LINE BRANCHLESS CALCULUS CORE ---
    m = m * LAMBDA - currentGradValue * GRAD_SCALE;
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
        // Acceleration positive: step up by exactly 1 token unit
        w += 1;
        m = m * 0.5; // Kinetic damping velocity wash-out
      } else if (isMomNegative && w > 0) {
        // Acceleration negative: step down by exactly 1 token unit
        w -= 1;
        m = m * 0.5;
      }
    }

    const realWeightFloat = w - 1; // Map unsigned token token back to mathematical float
    momentumHistory.push(currentMomentumInt);
    weightHistory.push(realWeightFloat);
  }

  // --- FIX: AUTOMATED CHARACTER-PADDED LAYOUT RENDERER ---
  // Dynamically analyzes the maximum label length to generate perfect X-axis text bounds
  const plotAndAlignAxis = (history: number[], height: number, color: string) => {
    const rawChartStr = chart.plot(history, { height, colors: [color] } as any);
    const chartLines = rawChartStr.split("\n");

    // Extract the absolute length of the dynamic Y-axis label text column row
    let maxYLabelLength = 0;
    for (let i = 0; i < chartLines.length; i++) {
      const match = chartLines[i].match(/^\s*[\d\.\-]+/);
      if (match && match[0].length > maxYLabelLength) {
        maxYLabelLength = match[0].length;
      }
    }

    // Generate perfect spacing margins natively
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
  console.log("VISUAL TRANSITION REPORT: CHANNELS ALIGNMENT DATA MAP     ");
  console.log("#########################################################");

  console.log("[1. INCOMING GRADIENTS FORCE]");
  plotAndAlignAxis(gradientHistory, 6, chart.red);

  console.log("[2. INTERNAL REGISTER SIGNED MOMENTUM]");
  plotAndAlignAxis(momentumHistory, 8, chart.cyan);

  console.log("[3. EXTREME ACTIVE TERNARY WEIGHTS POSITION (-1.0, 0.0, +1.0)]");
  plotAndAlignAxis(weightHistory, 6, chart.green);
  console.log("=========================================================");
}

runFusedWeightSimulation();

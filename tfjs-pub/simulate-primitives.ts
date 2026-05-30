import * as tf from "@tensorflow/tfjs";
import * as chart from "asciichart";

const STEPS = 100;
const GRAD_SCALE = 15.0; // The continuous gamma intake multiplier
const LAMBDA = 0.85; // The unconditional exponential time decay factor

function runStreamlinedSimulation() {
  console.log("=========================================================");
  console.log("LAUNCHING TWO-LINE BRANCHLESS WARP PRIMITIVE SIMULATION ");
  console.log(`Time Decay Friction : λ = ${LAMBDA}`);
  console.log(`Gradient Scale Force: γ = ${GRAD_SCALE}`);
  console.log("=========================================================");

  // Ingestion curves for raw gradient inputs
  const g1Curve: number[] = [];
  const g2Curve: number[] = [];
  const g3Curve: number[] = [];

  // Response curves for internal momentum states
  const m1Curve: number[] = [];
  const m2Curve: number[] = [];
  const m3Curve: number[] = [];

  // Continuous floating-point tracking registers (The Underlying Manifolds)
  let m1 = 0.0,
    m2 = 0.0,
    m3 = 0.0;

  for (let t = 0; t < STEPS; t++) {
    let g1 = 0.0,
      g2 = 0.0,
      g3 = 0.0;

    // --- SCENARIO DATA STREAMS ---
    // Scenario 1: Slow start -> Quick negative push -> Decay to gradient zero
    if (t >= 0 && t < 20) g1 = 0.01;
    if (t >= 20 && t < 50) g1 = 0.25;
    if (t >= 50 && t < 100) g1 = Math.max(0, 0.25 * (1.0 - (t - 50) / 30));

    // Scenario 2: Slow start -> Quick positive push -> Decay to gradient zero
    if (t >= 0 && t < 20) g2 = -0.01;
    if (t >= 20 && t < 50) g2 = -0.25;
    if (t >= 50 && t < 100) g2 = Math.min(0, -0.25 * (1.0 - (t - 50) / 30));

    // Scenario 3: Wavering direction shifts -> Decay -> RESIDUAL NOISE CRADLE HOLD
    if (t >= 0 && t < 20) g3 = -0.15;
    if (t >= 20 && t < 50) g3 = 0.2;
    if (t >= 50 && t < 80) g3 = Math.max(0, 0.2 * (1.0 - (t - 50) / 30));
    if (t >= 80 && t < 100) g3 = 0.03; // The active trailing residual noise shelf

    g1Curve.push(g1);
    g2Curve.push(g2);
    g3Curve.push(g3);

    // --- THE TWO-LINE BRANCHLESS ARITHMETIC CORE ENGINE ---
    // Line 1: Continuous momentum accumulation vector update (Negative gradient pushes positive)
    m1 = m1 * LAMBDA - g1 * GRAD_SCALE;
    m2 = m2 * LAMBDA - g2 * GRAD_SCALE;
    m3 = m3 * LAMBDA - g3 * GRAD_SCALE;

    // Bounded boundaries check to protect our strict 6-bit register tracking limits (-32 to 31)
    m1 = Math.max(-32.0, Math.min(31.0, m1));
    m2 = Math.max(-32.0, Math.min(31.0, m2));
    m3 = Math.max(-32.0, Math.min(31.0, m3));

    // Line 2: Stochastic Rounding Layer 1 Gate (Casts the continuous float back into integer space)
    const roundRegister = (val: number): number => {
      const floorVal = Math.floor(val);
      const fractionalPart = val - floorVal;
      return Math.random() < fractionalPart ? floorVal + 1 : floorVal;
    };

    // Collect the discrete register states for our visual line graphs
    m1Curve.push(roundRegister(m1));
    m2Curve.push(roundRegister(m2));
    m3Curve.push(roundRegister(m3));
  }

  // --- DISPLAY INTERFACE RENDERERS ---
  const plotPoints = 40;
  const downsample = (arr: number[]) => {
    const stepSize = Math.floor(arr.length / plotPoints);
    return Array.from({ length: plotPoints }, (_, i) => arr[i * stepSize]);
  };

  const renderXAxis = () => {
    const pad = "        ";
    return `${pad}└${"─".repeat(plotPoints - 1)}\n${pad}0` + " ".repeat(6) + "20" + " ".repeat(6) + "40" + " ".repeat(6) + "60" + " ".repeat(6) + "80" + " ".repeat(4) + "100 (Steps)";
  };

  console.log("\n#########################################################");
  console.log("SCENARIO 1: SMOOTH NEGATIVE TRAJECTORY MAPPING           ");
  console.log("#########################################################");
  console.log("[INCOMING GRADIENTS]");
  console.log(chart.plot(downsample(g1Curve), { height: 6, colors: [chart.red] } as any));
  console.log(renderXAxis());
  console.log("[RESULTING REGISTERS MOMENTUM]");
  console.log(chart.plot(downsample(m1Curve), { height: 8, colors: [chart.cyan] } as any));
  console.log(renderXAxis());

  console.log("\n#########################################################");
  console.log("SCENARIO 2: SMOOTH POSITIVE TRAJECTORY MAPPING           ");
  console.log("#########################################################");
  console.log("[INCOMING GRADIENTS]");
  console.log(chart.plot(downsample(g2Curve), { height: 6, colors: [chart.red] } as any));
  console.log(renderXAxis());
  console.log("[RESULTING REGISTERS MOMENTUM]");
  console.log(chart.plot(downsample(m2Curve), { height: 8, colors: [chart.yellow] } as any));
  console.log(renderXAxis());

  console.log("\n#########################################################");
  console.log("SCENARIO 3: DYNAMIC WAVERING & STABILIZATION ARENA       ");
  console.log("#########################################################");
  console.log("[INCOMING GRADIENTS]");
  console.log(chart.plot(downsample(g3Curve), { height: 6, colors: [chart.red] } as any));
  console.log(renderXAxis());
  console.log("[RESULTING REGISTERS MOMENTUM]");
  console.log(chart.plot(downsample(m3Curve), { height: 8, colors: [chart.magenta] } as any));
  console.log(renderXAxis());
  console.log("=========================================================");
}

runStreamlinedSimulation();

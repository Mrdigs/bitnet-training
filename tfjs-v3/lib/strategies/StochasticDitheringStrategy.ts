import * as tf from "@tensorflow/tfjs-node";
import { TwoBitSixBitAbstractStrategy } from "./TwoBitSixBitAbstractStrategy";

export interface StochasticDitheringConfig {
  gradScale?: number;
  K?: number;
  baseFriction?: number;
  maxFriction?: number;
  sensitivity?: number;
}

export class StochasticDitheringStrategy extends TwoBitSixBitAbstractStrategy {
  private readonly gradScale: number;
  private readonly K: number;
  private readonly baseFriction: number;
  private readonly maxFriction: number;
  private readonly sensitivity: number;
  private t: number = 0;

  private static readonly LUT_DATA = [0.0, 0.004, 0.005, 0.006, 0.007, 0.009, 0.012, 0.014, 0.018, 0.023, 0.028, 0.035, 0.044, 0.055, 0.069, 0.086, 0.107, 0.134, 0.168, 0.21, 0.262, 0.328, 0.41, 0.512, 0.64, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

  constructor(config: StochasticDitheringConfig = {}) {
    super();
    this.gradScale = config.gradScale ?? 30.0;
    this.K = config.K ?? 0.5;
    this.baseFriction = config.baseFriction ?? 0.75;
    this.maxFriction = config.maxFriction ?? 1.0;
    this.sensitivity = config.sensitivity ?? 1.5;
  }

  protected override getInitialSixBitState(units: number, inFeatures: number): tf.Tensor {
    return tf.zeros([units, inFeatures], "int32");
  }

  protected override optimizeSixBitElement(currentWeight: number, currentSixBitState: number, gradient: number, lr: number): { newWeight: number; newSixBitState: number } {
    this.t += 1;

    // SECTION 1: TWO'S COMPLEMENT SIGN RECONSTRUCTION
    let momentum = currentSixBitState;
    if (momentum > 31) {
      momentum -= 64;
    }

    // SECTION 2: PROPORTIONAL THERMODYNAMIC FRICTION FLYWHEEL
    const absGrad = Math.abs(gradient);
    const rawLambda = this.baseFriction + absGrad * this.sensitivity;
    const dynamicLambda = Math.max(this.baseFriction, Math.min(this.maxFriction, rawLambda));

    const continuousNextMom = momentum * dynamicLambda - gradient * this.gradScale;
    let nextMomentum = Math.max(-31, Math.min(31, continuousNextMom));

    // SECTION 3: STOCHASTIC DITHERING / STOCHASTIC FLOOR ROUNDING
    const floorMom = Math.floor(nextMomentum);
    const fractionalPart = nextMomentum - floorMom;
    if (Math.random() < fractionalPart) {
      nextMomentum = Math.ceil(nextMomentum);
    } else {
      nextMomentum = floorMom;
    }

    // SECTION 4: STOCHASTIC ANNEALING LUT GATE
    const absVelocity = Math.abs(nextMomentum);
    const scale_t = 1.0 + this.K * (1.0 - lr);
    const annealedIndex = Math.max(0, Math.min(32, Math.floor(absVelocity * scale_t)));
    const flipProbability = StochasticDitheringStrategy.LUT_DATA[annealedIndex];

    // SECTION 5: ZERO-CROSSING FILTER AND QUANTIZATION FLIP
    let newWeight = currentWeight;

    if (Math.random() < flipProbability) {
      const isMomPositive = nextMomentum > 0;
      const isMomNegative = nextMomentum < 0;

      const canIncrement = newWeight === -1 || newWeight === 0;
      const canDecrement = newWeight === 1 || newWeight === 0;

      if (isMomPositive && canIncrement) {
        newWeight += 1;
        nextMomentum = Math.floor(nextMomentum * 0.5);
      } else if (isMomNegative && canDecrement) {
        newWeight -= 1;
        nextMomentum = Math.floor(nextMomentum * 0.5);
      }
    }

    // SECTION 6: PACKING TRANSITION
    let unsignedSixBitState = nextMomentum;
    if (unsignedSixBitState < 0) {
      unsignedSixBitState += 64;
    }

    return {
      newWeight,
      newSixBitState: unsignedSixBitState,
    };
  }
}

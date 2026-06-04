import { runEvaluationHarness, BenchmarkCandidate } from "./benchmark-mnist";
import { StochasticDitheringStrategy } from "./lib/strategies/StochasticDitheringStrategy";
import { UncompromisedQatAdamStrategy } from "./lib/strategies/UncompromisedQatAdamStrategy";

const ditheringCandidate: BenchmarkCandidate = {
  label: "Stochastic Dithering (8-bit)",
  learningRate: 0.01,
  // The strategy constructor requires zero shape attributes, so we ignore the factory arguments!
  factory: () =>
    new StochasticDitheringStrategy({
      gradScale: 30.0,
      K: 0.5,
      baseFriction: 0.75,
      maxFriction: 1.0,
      sensitivity: 1.5,
    }),
};

const qatAdamControl: BenchmarkCandidate = {
  label: "Full-Precision QAT Adam Control",
  learningRate: 0.001,
  // The uncompromised Adam strategy explicitly requires shape parameters to allocate
  // its hidden tracking floats, so the factory handles passing them down cleanly!
  factory: (units, inFeatures) => new UncompromisedQatAdamStrategy(units, inFeatures),
};

runEvaluationHarness(ditheringCandidate, qatAdamControl);

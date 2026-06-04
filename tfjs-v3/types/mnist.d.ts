declare module "mnist" {
  export interface MnistSample {
    input: number[];
    output: number[];
  }

  export interface MnistDataset {
    training: MnistSample[];
    test: MnistSample[];
  }

  export function set(trainingCount: number, testCount: number): MnistDataset;
}

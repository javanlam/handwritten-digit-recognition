export interface PredictionResult {
  predicted_class: number;
  probability: number;
  top3_classes: number[];
  top3_probabilities: number[];
}

export interface GridData {
  rows: number;
  cols: number;
  data: number[][];
}
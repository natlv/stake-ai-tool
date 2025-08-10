export const BASELINES: Record<string, Partial<{
  grossProfitMargin: number; netProfitMargin: number; operatingProfitMargin: number;
  peRatio: number; debtToEquity: number; freeCashFlowRatio: number; operatingCashFlow: number;
}>> = {
  ABC: { grossProfitMargin: 0.38, peRatio: 22.5, debtToEquity: 1.1 },
  XYZ: { netProfitMargin: 0.12, operatingProfitMargin: 0.18, debtToEquity: 0.6 },
};

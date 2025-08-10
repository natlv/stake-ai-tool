export type Extracted = {
  documentType: string | null;
  stockName: string | null;
  highlights: string[];
  ratios: {
    grossProfitMargin?: number | null;
    netProfitMargin?: number | null;
    operatingProfitMargin?: number | null;
    peRatio?: number | null;
    debtToEquity?: number | null;
    freeCashFlowRatio?: number | null;
    operatingCashFlow?: number | null;
  };
  techIndicators: {
    sma50?: number[] | null; // Seq
    sma200?: number[] | null;
    macd?: number[] | null;
    rsi?: number[] | null;
    vix?: number[] | null;
  };
  macro: {
    gdp?: number | null;
    interestRates?: number | null;
    cpi?: number | null;
  };
  industry: {
    gics?: string | null;
    embedding?: number[] | null;
    sector?:
      | "Energy"
      | "Materials"
      | "Industrials"
      | "Consumer Discretionary"
      | "Consumer Staples"
      | "Health Care"
      | "Financials"
      | "IT"
      | "Communication Services"
      | "Utilities"
      | "Real Estate"
      | null;
  };
  marketSentimentCategory:
    | "Very Bearish"
    | "Bearish"
    | "Neutral"
    | "Bullish"
    | "Very Bullish";
  anomalies: string[]; // potential risks
  decision: "BUY" | "SELL";
  confidenceGrade: "A" | "A-" | "B+" | "B" | "B-" | "C+" | "C" | "D";
};

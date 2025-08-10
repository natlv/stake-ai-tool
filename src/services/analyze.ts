import type { Extracted } from "@/types/extracted";

function delay(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

function mockExtracted(): Extracted {
  return {
    documentType: "10-K",
    stockName: "JP Morgan",
    highlights: [
      "Record revenue growth",
      "Healthy cash position",
      "Operational efficiency gains",
    ],
    ratios: {
      grossProfitMargin: 0.52,
      netProfitMargin: 0.18,
      operatingProfitMargin: 0.22,
      peRatio: 18.7,
      debtToEquity: 0.45,
      freeCashFlowRatio: 0.12,
      operatingCashFlow: 1250000000,
    },
    techIndicators: {
      sma50: [100, 101, 102, 103, 104, 105, 106],
      sma200: [90, 91, 92, 93, 94, 95, 96],
      macd: [0.2, 0.3, 0.25, 0.4, 0.35],
      rsi: [45, 52, 60, 58, 55],
      vix: [16, 17, 15, 14, 13],
    },
    macro: {
      gdp: 2.4,
      interestRates: 4.75,
      cpi: 3.1,
    },
    industry: {
      gics: "45102010",
      embedding: [0.12, 0.04, -0.33, 0.88, -0.1],
      sector: "IT",
    },
    marketSentimentCategory: "Bullish",
    anomalies: ["Uncertain economic outlook", "Decreasing YoY revenue"],
    decision: "BUY",
    confidenceGrade: "B+",
  };
}

export async function analyzePdf(file: File): Promise<Extracted> {
  // In Next.js, you'd POST FormData to /api/analyze. Here we mock it.
  // const fd = new FormData();
  // fd.append("file", file);
  await delay(1500);
  return mockExtracted();
}

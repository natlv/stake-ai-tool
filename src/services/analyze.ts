import type { Extracted } from "@/types/extracted";
import { loadTechIndicatorsForTicker } from "@/services/features";

function delay(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

function mockExtracted(): Extracted {
  return {
    documentType: "Financial Report",
    stockName: "JP Morgan",
    highlights: [
      "Increasing dividend payout",
      "Healthy cash position",
      "Operational efficiency gains",
    ],
    ratios: {
      grossProfitMargin: null,
      netProfitMargin: 0.32,
      operatingProfitMargin: 0.45,
      peRatio: 12.8,
      debtToEquity: 1.25,
      freeCashFlowRatio: 0.08,
      operatingCashFlow: 85000000000,
    },
    techIndicators: {
      sma50: [100, 101, 102, 103, 104, 105, 106],
      sma200: [90, 91, 92, 93, 94, 95, 96],
      macd: [0.2, 0.3, 0.25, 0.4, 0.35],
      rsi: [45, 52, 60, 58, 55],
      vix: [16, 17, 15, 14, 13],
    },
    macro: {
      gdp: 2.1,
      interestRates: 5.25,
      cpi: 3.2,
    },
    industry: {
      gics: "40101020",
      embedding: [0.12, 0.04, -0.33, 0.88, -0.1],
      sector: "Financials",
    },
    marketSentimentCategory: "Bullish",
    anomalies: ["Uncertain economic outlook", "Decreasing YoY revenue"],
    decision: "BUY",
    confidenceGrade: "B+",
  };
}

export async function analyzePdf(file: File): Promise<Extracted> {
  // In production, post to Stage 2 inference; here we enrich with Parquet if available.
  await delay(800);
  const base = mockExtracted();
  try {
    const tech = await loadTechIndicatorsForTicker("JPM");
    if (tech) {
      base.techIndicators = {
        ...base.techIndicators,
        ...tech,
      };
    }
  } catch {}
  return base;
}

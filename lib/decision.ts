import type { Extracted } from "@/app/api/analyze/route";

const NUMERIC_FIELDS: (keyof NonNullable<Extracted["ratios"]>)[] = [
  "grossProfitMargin",
  "netProfitMargin",
  "operatingProfitMargin",
  "peRatio",
  "debtToEquity",
  "freeCashFlowRatio",
  "operatingCashFlow",
];

export function anomaliesFromText(
  text: string,
  extracted: { ratios?: Extracted["ratios"] },
  baselinesForTicker: Partial<Record<keyof Extracted["ratios"], number>>
): string[] {
  const out: string[] = [];
  const t = text.toLowerCase();
  const redFlags = [
    "material weakness",
    "going concern",
    "impairment",
    "restatement",
    "covenant breach",
  ];
  for (const p of redFlags) if (t.includes(p)) out.push(p);

  const ratios = extracted.ratios || {};
  for (const key of NUMERIC_FIELDS) {
    const docVal = ratios[key];
    const baseVal = (baselinesForTicker as any)[key];
    if (docVal != null && baseVal != null && Number.isFinite(docVal) && Number.isFinite(baseVal)) {
      const diff = Math.abs((Number(docVal) - Number(baseVal)) / Number(baseVal));
      if (diff > 0.1) out.push(`${key} deviates >10% vs cache`);
    }
  }
  return out;
}

export function decide(extracted: Extracted): "BUY" | "SELL" {
  const sentiment = extracted.marketSentimentCategory;
  const anomalies = extracted.anomalies?.length || 0;
  const r = extracted.ratios || {};
  const peOk = typeof r.peRatio === "number";
  const deOk = typeof r.debtToEquity === "number" && (r.debtToEquity as number) < 2;
  if ((sentiment === "Bullish" || sentiment === "Very Bullish") && anomalies < 2 && peOk && deOk) {
    return "BUY";
  }
  return "SELL";
}

export function confidenceGrade(extracted: Extracted): "A" | "A-" | "B+" | "B" | "B-" | "C+" | "C" | "D" {
  let count = 0;
  const r = extracted.ratios || {};
  const m = extracted.macro || {};
  const ind = extracted.industry || {};
  const tech = extracted.techIndicators || {};

  for (const k of Object.keys(r)) if ((r as any)[k] != null) count++;
  for (const k of Object.keys(m)) if ((m as any)[k] != null) count++;
  for (const k of Object.keys(ind)) if ((ind as any)[k] != null) count++;
  for (const k of Object.keys(tech)) if ((tech as any)[k] != null) count++;

  if (count >= 10) return "A";
  if (count >= 9) return "A-";
  if (count >= 8) return "B+";
  if (count >= 6) return "B";
  if (count >= 5) return "B-";
  if (count >= 4) return "C+";
  if (count >= 3) return "C";
  return "D";
}

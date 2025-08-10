// Finance-weighted sentiment classifier

const POSITIVE = [
  { w: "raise guidance", s: 3 },
  { w: "record", s: 2 },
  { w: "beat", s: 2 },
  { w: "expansion", s: 2 },
  { w: "accelerat", s: 2 },
  { w: "strong", s: 1 },
  { w: "improv", s: 1 },
  { w: "grow", s: 1 },
  { w: "profit", s: 1 },
  { w: "cash flow", s: 1 },
];

const NEGATIVE = [
  { w: "going concern", s: -5 },
  { w: "material weakness", s: -4 },
  { w: "liquidity crunch", s: -4 },
  { w: "covenant breach", s: -3 },
  { w: "restatement", s: -3 },
  { w: "impairment", s: -3 },
  { w: "miss", s: -2 },
  { w: "declin", s: -1 },
  { w: "downgrade", s: -2 },
  { w: "slowdown", s: -1 },
];

export function marketSentimentCategory(text: string): "Very Bearish" | "Bearish" | "Neutral" | "Bullish" | "Very Bullish" {
  const t = text.toLowerCase();
  let score = 0;
  for (const { w, s } of POSITIVE) if (t.includes(w)) score += s;
  for (const { w, s } of NEGATIVE) if (t.includes(w)) score += s;
  // Normalize a bit by length
  const lenAdj = Math.max(0, Math.min(3, Math.floor(t.length / 20000)));
  score = score - lenAdj; // longer docs dilute signal slightly

  if (score <= -5) return "Very Bearish";
  if (score <= -2) return "Bearish";
  if (score < 2) return "Neutral";
  if (score < 5) return "Bullish";
  return "Very Bullish";
}

// Parsing helpers for financial documents

export function detectDocumentType(text: string): "10-K" | "10-Q" | "8-K" | "earnings call" | "press release" | "annual report" | null {
  const t = text.toLowerCase();
  if (/\b10\s*-\s*k\b|form\s+10-?k/i.test(text)) return "10-K";
  if (/\b10\s*-\s*q\b|form\s+10-?q/i.test(text)) return "10-Q";
  if (/\b8\s*-\s*k\b|form\s+8-?k/i.test(text)) return "8-K";
  if (/earnings\s+call|prepared\s+remarks|q&?a\s+session/i.test(text)) return "earnings call";
  if (/press\s+release|newswire|pr\s*news/i.test(text)) return "press release";
  if (/annual\s+report|fy\s*\d{2}/i.test(text)) return "annual report";
  return null;
}

export function detectStockName(text: string): string | null {
  const firstBlock = text.slice(0, 2000);
  const exTicker = firstBlock.match(/\b(?:NASDAQ|NYSE|AMEX|TSX|LSE)\s*[:\-]\s*([A-Z]{1,5})\b/);
  const parenTicker = firstBlock.match(/\(([A-Z]{1,5})\)/);
  const ticker = (exTicker?.[1] || parenTicker?.[1])?.toUpperCase() || null;
  // Company/title heuristics: first prominent uppercase line
  const lines = firstBlock.split(/\n+/).map((l) => l.trim());
  const titleLine = lines.find((l) => /[A-Za-z]/.test(l) && l.length > 3 && (l === l.toUpperCase() || /Inc\.|Corp\.|Corporation|Ltd\.|PLC/i.test(l))) || "";
  const company = titleLine.replace(/\s{2,}/g, " ").replace(/^\d+\.?\s*/, "").trim();
  if (company && ticker) return `${company} (${ticker})`;
  if (company) return company;
  if (ticker) return ticker;
  return null;
}

export function extractHighlights(text: string, n = 5): string[] {
  const firstFewPages = text.split("\f").slice(0, 3).join(" ");
  const corpus = firstFewPages.length > 200 ? firstFewPages : text;
  const sentences = corpus
    .replace(/\s+/g, " ")
    .split(/(?<=[\.!?])\s+/)
    .filter((s) => s.length > 20 && s.length < 300);
  const positive = ["record", "growth", "beat", "raise guidance", "expansion", "profit", "cash flow", "revenue", "margin"];
  const negative = ["miss", "decline", "impairment", "downgrade", "weakness", "going concern", "liquidity crunch", "restatement"];
  const score = (s: string) => {
    const t = s.toLowerCase();
    let sc = 0;
    for (const w of positive) if (t.includes(w)) sc += 2;
    for (const w of negative) if (t.includes(w)) sc -= 3;
    return sc + Math.min(2, Math.floor(s.length / 120));
  };
  return sentences
    .map((s) => ({ s: s.trim().replace(/^[-•\u2022]+\s*/, ""), score: score(s) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, n)
    .map(({ s }) => s);
}

function windowAround(text: string, idx: number, before = 80, after = 80) {
  return text.slice(Math.max(0, idx - before), idx + after);
}

function parsePercentToDecimal(raw: string): number {
  const val = Number(raw.replace(/%/, "").trim());
  return Number.isFinite(val) ? val / 100 : NaN;
}

function extractNear(text: string, keyRegex: RegExp): { percent?: number; decimal?: number; currency?: number } | null {
  const m = keyRegex.exec(text);
  if (!m) return null;
  const win = windowAround(text, m.index, 120, 120);
  const pct = win.match(/-?\d{1,3}(?:\.\d+)?\s?%/);
  if (pct) return { percent: parsePercentToDecimal(pct[0]) };
  const dec = win.match(/-?\d+(?:\.\d+)?x?/i);
  if (dec) return { decimal: Number(dec[0].replace(/x/i, "")) };
  const cur = win.match(/\$?\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s*(million|billion|thousand|m|bn|k))?/i);
  if (cur) {
    const base = Number(cur[1].replace(/,/g, ""));
    const unit = (cur[2] || "").toLowerCase();
    const mult = unit === "billion" || unit === "bn" ? 1e9 : unit === "million" || unit === "m" ? 1e6 : unit === "thousand" || unit === "k" ? 1e3 : 1;
    return { currency: base * mult };
  }
  return null;
}

export function extractRatios(text: string) {
  const out: {
    grossProfitMargin?: number | null;
    netProfitMargin?: number | null;
    operatingProfitMargin?: number | null;
    peRatio?: number | null;
    debtToEquity?: number | null;
    freeCashFlowRatio?: number | null;
    operatingCashFlow?: number | null;
  } = {};

  const specs: Array<{
    key: keyof typeof out;
    regex: RegExp;
    kind: "percent" | "decimal" | "currency" | "either";
  }> = [
    { key: "grossProfitMargin", regex: /gross\s+profit\s+margin|gross\s+margin/i, kind: "either" },
    { key: "netProfitMargin", regex: /net\s+profit\s+margin|net\s+margin/i, kind: "either" },
    { key: "operatingProfitMargin", regex: /operating\s+(profit\s+)?margin/i, kind: "either" },
    { key: "peRatio", regex: /\bP\/?E\b|price\s*to\s*earnings/i, kind: "decimal" },
    { key: "debtToEquity", regex: /debt\s*[:\/]?\s*to\s*\s*equity|D\/?E\b|debt-?to-?equity/i, kind: "decimal" },
    { key: "freeCashFlowRatio", regex: /free\s+cash\s+flow\s+ratio|FCF\s*ratio/i, kind: "either" },
    { key: "operatingCashFlow", regex: /operating\s+cash\s+flow|OCF\b/i, kind: "currency" },
  ];

  for (const s of specs) {
    const got = extractNear(text, s.regex);
    if (!got) continue;
    let val: number | null = null;
    if (s.kind === "percent" && typeof got.percent === "number") val = got.percent;
    else if (s.kind === "decimal" && typeof got.decimal === "number") val = got.decimal;
    else if (s.kind === "currency" && typeof got.currency === "number") val = got.currency;
    else if (s.kind === "either") val = (got.percent ?? got.decimal) ?? null;
    (out as any)[s.key] = val;
  }

  return out;
}

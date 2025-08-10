import pdf from "pdf-parse";
import { NextResponse } from "next/server";
import { detectDocumentType, detectStockName, extractHighlights, extractRatios } from "@/lib/parse";
import { marketSentimentCategory } from "@/lib/sentiment";
import { BASELINES } from "@/lib/financebenchCache";
import { anomaliesFromText, decide, confidenceGrade } from "@/lib/decision";

export const runtime = "nodejs";

// Local type definition to keep route self-contained
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
    sma50?: number[] | null;
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
  marketSentimentCategory: "Very Bearish" | "Bearish" | "Neutral" | "Bullish" | "Very Bullish";
  anomalies: string[];
  decision: "BUY" | "SELL";
  confidenceGrade: "A" | "A-" | "B+" | "B" | "B-" | "C+" | "C" | "D";
  looksImageHeavy?: boolean;
  imageHeavyNote?: string | null;
};

function toBuffer(file: File): Promise<Buffer> {
  return file.arrayBuffer().then((ab) => Buffer.from(ab));
}

function splitPagesText(text: string): string[] {
  // pdf-parse separates pages with form-feed (\f). Fallback to heuristic if missing.
  if (text.includes("\f")) return text.split("\f");
  // Heuristic split: look for "\n\n\x0c" or multiple newlines followed by Page
  return text.split(/\n\s*Page \d+\s*\n|\n{3,}/g);
}

// Simple numeric extractors for extra sections
const numRe = /-?\d{1,3}(?:,\d{3})*(?:\.\d+)?/g;
const pctRe = /-?\d{1,3}(?:\.\d+)?\s?%/g;

function parseSequenceNearby(source: string, label: RegExp, max = 12): number[] | null {
  const m = label.exec(source);
  if (!m) return null;
  const idx = m.index;
  const window = source.slice(Math.max(0, idx - 200), idx + 400);
  const arrMatch = window.match(/\[\s*([-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?){2,})\s*\]/);
  if (arrMatch) {
    return arrMatch[1]
      .split(/\s*,\s*/)
      .map((x) => Number(x))
      .filter((x) => Number.isFinite(x))
      .slice(0, max);
  }
  const nums = window.match(/[-+]?\d+(?:\.\d+)?/g);
  if (!nums) return null;
  const parsed = nums.map((n) => Number(n)).filter((x) => Number.isFinite(x));
  return parsed.length >= 3 ? parsed.slice(0, max) : null;
}

function parsePercentNearby(source: string, label: RegExp): number | null {
  const m = label.exec(source);
  if (!m) return null;
  const idx = m.index;
  const window = source.slice(Math.max(0, idx - 120), idx + 200);
  const pct = window.match(pctRe)?.[0];
  if (pct) return Number(pct.replace(/%/, " ")) / 100;
  const dec = window.match(/-?\d+(?:\.\d+)?/);
  return dec ? Number(dec[0]) : null;
}

function parseCurrencyNearby(source: string, label: RegExp): number | null {
  const m = label.exec(source);
  if (!m) return null;
  const idx = m.index;
  const window = source.slice(Math.max(0, idx - 120), idx + 200);
  const raw = window.match(/\$?\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s*(million|billion|thousand|m|bn|k))?/i);
  if (!raw) return null;
  const base = Number(raw[1].replace(/,/g, ""));
  if (!Number.isFinite(base)) return null;
  const unit = (raw[2] || "").toLowerCase();
  const mult = unit === "billion" || unit === "bn" ? 1e9 : unit === "million" || unit === "m" ? 1e6 : unit === "thousand" || unit === "k" ? 1e3 : 1;
  return base * mult;
}

function parseTechIndicators(text: string) {
  return {
    sma50: parseSequenceNearby(text, /SMA\s*50|SMA50|50\s*-?\s*day\s+SMA/i),
    sma200: parseSequenceNearby(text, /SMA\s*200|SMA200|200\s*-?\s*day\s+SMA/i),
    macd: parseSequenceNearby(text, /MACD/i),
    rsi: parseSequenceNearby(text, /RSI/i),
    vix: parseSequenceNearby(text, /VIX/i),
  } as Extracted["techIndicators"];
}

function parseMacro(text: string) {
  const gdp = parsePercentNearby(text, /\bGDP\b/i);
  const interestRates = parsePercentNearby(text, /interest\s*rate[s]?/i);
  const cpi = parsePercentNearby(text, /\bCPI\b|consumer\s+price\s+index/i);
  return { gdp, interestRates, cpi } as Extracted["macro"];
}

const SECTORS = [
  "Energy",
  "Materials",
  "Industrials",
  "Consumer Discretionary",
  "Consumer Staples",
  "Health Care",
  "Financials",
  "IT",
  "Communication Services",
  "Utilities",
  "Real Estate",
] as const;

function parseIndustry(text: string) {
  const gicsMatch = text.match(/\b\d{8}\b/);
  const sector = SECTORS.find((s) => new RegExp(`\\b${s.replace(/\s+/g, "\\s+") }\\b`, "i").test(text)) || null;
  const embedding = null; // Only populate if explicitly present; omitting generation.
  const gics = gicsMatch ? gicsMatch[0] : null;
  return { gics, embedding, sector } as Extracted["industry"];
}

function deriveTicker(text: string): string | null {
  const m1 = text.match(/\b(?:NASDAQ|NYSE|AMEX|TSX|LSE)\s*[:\-]\s*([A-Z]{1,5})\b/);
  if (m1) return m1[1].toUpperCase();
  const m2 = text.match(/\(([A-Z]{1,5})\)/);
  if (m2) return m2[1].toUpperCase();
  const m3 = text.match(/\bTicker\s*[:\-]\s*([A-Z]{1,5})\b/i);
  if (m3) return m3[1].toUpperCase();
  return null;
}

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");
    if (!(file instanceof File)) {
      return NextResponse.json({ error: "Missing 'file'" }, { status: 400 });
    }
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      return NextResponse.json({ error: "Only PDF is supported" }, { status: 415 });
    }

    const buffer = await toBuffer(file);
    const parsed = await pdf(buffer);
    const rawText = parsed.text || "";
    const pages = splitPagesText(rawText);
    const totalChars = rawText.length;
    const charsPerPage = pages.map((p) => p.length);
    const looksImageHeavy = totalChars < 1000 || (charsPerPage.length > 0 && charsPerPage.every((x) => x < 300));

    const text = rawText;

    const documentType = detectDocumentType(text);
    const stockName = detectStockName(text);
    const ratios = extractRatios(text);
    const techIndicators = parseTechIndicators(text);
    const macro = parseMacro(text);
    const industry = parseIndustry(text);
    const sentiment = marketSentimentCategory(text);

    const ticker = deriveTicker(text);
    const baselines = ticker ? BASELINES[ticker] || {} : {};

    const extracted: Extracted = {
      documentType,
      stockName,
      highlights: extractHighlights(text, 5),
      ratios: {
        grossProfitMargin: ratios.grossProfitMargin ?? null,
        netProfitMargin: ratios.netProfitMargin ?? null,
        operatingProfitMargin: ratios.operatingProfitMargin ?? null,
        peRatio: ratios.peRatio ?? null,
        debtToEquity: ratios.debtToEquity ?? null,
        freeCashFlowRatio: ratios.freeCashFlowRatio ?? null,
        operatingCashFlow: ratios.operatingCashFlow ?? null,
      },
      techIndicators: {
        sma50: techIndicators.sma50 ?? null,
        sma200: techIndicators.sma200 ?? null,
        macd: techIndicators.macd ?? null,
        rsi: techIndicators.rsi ?? null,
        vix: techIndicators.vix ?? null,
      },
      macro: {
        gdp: macro.gdp ?? null,
        interestRates: macro.interestRates ?? null,
        cpi: macro.cpi ?? null,
      },
      industry: {
        gics: industry.gics ?? null,
        embedding: industry.embedding ?? null,
        sector: industry.sector ?? null,
      },
      marketSentimentCategory: sentiment,
      anomalies: anomaliesFromText(text, { ratios }, baselines),
      decision: "SELL", // placeholder; will set after
      confidenceGrade: "D", // placeholder; will set after
      looksImageHeavy,
      imageHeavyNote: looksImageHeavy
        ? "This PDF appears image-heavy; text extraction may be incomplete. Consider uploading the source or enabling OCR."
        : null,
    };

    extracted.decision = decide({
      ...extracted,
    });
    extracted.confidenceGrade = confidenceGrade(extracted);

    return NextResponse.json(extracted);
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || "Failed to analyze PDF" }, { status: 500 });
  }
}

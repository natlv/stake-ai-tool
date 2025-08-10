import type { Extracted } from "@/types/extracted";
import { withConnection } from "@/services/duckdb";

const LAST_N = 64; // limit for UI sparklines

function pickCol(row: any, ...keys: string[]) {
  for (const k of keys) {
    if (row[k] != null) return row[k];
    const lower = k.toLowerCase();
    if (row[lower] != null) return row[lower];
  }
  return undefined;
}

export async function loadTechIndicatorsForTicker(
  ticker: string
): Promise<Partial<Extracted["techIndicators"]> | null> {
  const sym = ticker.toUpperCase();
  const url = `/models/panel_${sym}.parquet`;

  try {
    const data = await withConnection(async (conn) => {
      const q = await conn.query(
        `SELECT * FROM read_parquet('${url}')`
      );
      // Convert to plain JS rows
      const rows: any[] = q.toArray();
      if (!rows?.length) return null;
      // Keep chronological (assumes file already ordered); take last N
      const last = rows.slice(-LAST_N);

      const sma50 = last
        .map((r) => Number(pickCol(r, "SMA50")))
        .filter((x) => Number.isFinite(x));
      const sma200 = last
        .map((r) => Number(pickCol(r, "SMA200")))
        .filter((x) => Number.isFinite(x));
      const macd = last
        .map((r) => Number(pickCol(r, "MACD")))
        .filter((x) => Number.isFinite(x));
      const rsi = last
        .map((r) => Number(pickCol(r, "RSI")))
        .filter((x) => Number.isFinite(x));
      const vix = last
        .map((r) => Number(pickCol(r, "VIX")))
        .filter((x) => Number.isFinite(x));

      // If SMA columns missing, try to compute from Close if available
      const close = last
        .map((r) => Number(pickCol(r, "C", "Close")))
        .filter((x) => Number.isFinite(x));
      const ensuredSMA50 = sma50.length ? sma50 : simpleSMA(close, 50).slice(-LAST_N);
      const ensuredSMA200 = sma200.length ? sma200 : simpleSMA(close, 200).slice(-LAST_N);

      return {
        sma50: ensuredSMA50.length ? ensuredSMA50 : undefined,
        sma200: ensuredSMA200.length ? ensuredSMA200 : undefined,
        macd: macd.length ? macd : undefined,
        rsi: rsi.length ? rsi : undefined,
        vix: vix.length ? vix : undefined,
      } as Partial<Extracted["techIndicators"]>;
    });

    return data;
  } catch (err) {
    console.warn("Parquet fallback unavailable:", err);
    return null;
  }
}

// --- Simple indicator helpers ---
function simpleSMA(series: number[], window: number): number[] {
  if (!series?.length || window <= 1) return series || [];
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < series.length; i++) {
    sum += series[i];
    if (i >= window) sum -= series[i - window];
    if (i >= window - 1) out.push(sum / window);
  }
  return out;
}

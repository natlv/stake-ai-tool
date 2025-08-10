import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { Extracted } from "@/types/extracted";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Sparkline from "@/components/Sparkline";
import BadgeDecision from "@/components/BadgeDecision";
import GradePill from "@/components/GradePill";
import FieldRow from "@/components/FieldRow";

function formatNumber(n: number | null | undefined, opts?: Intl.NumberFormatOptions) {
  if (n === null || n === undefined || Number.isNaN(n)) return "Not present in document";
  return new Intl.NumberFormat(undefined, opts).format(n);
}

function formatPercent(n: number | null | undefined) {
  if (n === null || n === undefined || Number.isNaN(n)) return "Not present in document";
  const val = Math.abs(n) <= 1 ? n * 100 : n; // handle 0-1 or already %
  return `${val.toFixed(2)}%`;
}

const Results: React.FC = () => {
  const { state } = useLocation();
  const navigate = useNavigate();
  const result: Extracted | undefined = state?.result;

  React.useEffect(() => {
    document.title = result?.stockName ? `${result.stockName} Analysis — Results` : "Document Analysis Results";
  }, [result?.stockName]);

  const pillBase =
    "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset transition-colors";
  const toneTokenForSentiment = (
    s: Extracted["marketSentimentCategory"]
  ): "brand-green" | "warning" | "destructive" => {
    if (s === "Neutral") return "warning";
    if (s.includes("Bullish")) return "brand-green";
    return "destructive";
  };
  const pillClassesForToken = (token: "brand-green" | "warning" | "destructive") =>
    token === "destructive"
      ? "bg-[hsl(var(--destructive)_/_0.14)] text-[hsl(var(--destructive))] ring-[hsl(var(--destructive)_/_0.35)] hover:bg-[hsl(var(--destructive)_/_0.22)]"
      : token === "warning"
      ? "bg-[hsl(var(--warning)_/_0.14)] text-[hsl(var(--warning))] ring-[hsl(var(--warning)_/_0.35)] hover:bg-[hsl(var(--warning)_/_0.22)]"
      : "bg-[hsl(var(--brand-green)_/_0.14)] text-[hsl(var(--brand-green))] ring-[hsl(var(--brand-green)_/_0.35)] hover:bg-[hsl(var(--brand-green)_/_0.22)]";

  if (!result) {
    return (
      <main>
        <section className="container mx-auto min-h-[60vh] px-6 py-16 text-center">
          <h1 className="mb-4 text-3xl font-bold">No analysis found</h1>
          <p className="mb-6 text-muted-foreground">Start by uploading a PDF to analyze.</p>
          <Button variant="hero" onClick={() => navigate("/")}>Go back</Button>
        </section>
      </main>
    );
  }

  return (
    <main>
      <section className="container mx-auto px-6 py-10">
        <header className="mx-auto mb-8 max-w-4xl text-center">
          <h1 className="mb-3 text-4xl font-bold tracking-tight">Analysis Results</h1>
          <p className="text-muted-foreground">Document type and detected stock name are shown below with overall decision.</p>
        </header>

        {result.looksImageHeavy && (
          <div className="mx-auto mb-6 max-w-5xl rounded-xl border bg-secondary/40 px-4 py-3 text-sm text-muted-foreground ring-1 ring-inset ring-border">
            {result.imageHeavyNote || "This file appears image-heavy. OCR is disabled in this demo."}
          </div>
        )}

        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-3">
          <Card className="md:col-span-2 rounded-2xl transition hover:shadow-lg hover:shadow-[var(--shadow-glow)]">
            <CardHeader>
              <CardTitle>Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <FieldRow label="Document Type" value={result.documentType ?? null} />
                <FieldRow label="Detected Stock" value={result.stockName ?? null} />
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="flex items-center gap-3">
                  <span className="text-sm text-muted-foreground">Decision</span>
                  <BadgeDecision value={result.decision} />
                </div>
                <div className="flex items-center gap-3 -ml-2">
                  <span className="text-sm text-muted-foreground">Confidence Grade</span>
                  <GradePill grade={result.confidenceGrade as any} />
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-sm text-muted-foreground">Overall Sentiment</span>
                  <span
                    className={`${pillBase} ${pillClassesForToken(toneTokenForSentiment(result.marketSentimentCategory))}`}
                    aria-label={`Overall sentiment: ${result.marketSentimentCategory}`}
                  >
                    {result.marketSentimentCategory}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-2xl transition hover:shadow-lg hover:shadow-[var(--shadow-glow)]">
            <CardHeader>
              <CardTitle>Cool highlights</CardTitle>
            </CardHeader>
            <CardContent>
              {result.highlights?.length ? (
                <ul className="list-disc space-y-2 pl-5">
                  {result.highlights.map((h, i) => (
                    <li key={i} className="text-sm text-muted-foreground">{h}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">Not present in document</p>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="mx-auto mt-6 grid max-w-5xl grid-cols-1 gap-6">
          <Card className="rounded-2xl transition hover:shadow-lg hover:shadow-[var(--shadow-glow)]">
            <CardHeader>
              <CardTitle>Anomalies</CardTitle>
            </CardHeader>
            <CardContent>
              {result.anomalies && result.anomalies.length > 0 ? (
                <ul className="list-disc space-y-2 pl-5 text-sm text-muted-foreground">
                  {result.anomalies.map((a, i) => (
                    <li key={i}>{a}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">Not present in document</p>
              )}
            </CardContent>
          </Card>

          <Card className="rounded-2xl transition hover:shadow-lg hover:shadow-[var(--shadow-glow)]">
            <CardHeader>
              <CardTitle>Ratios</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <FieldRow label="Gross Profit Margin" value={result.ratios.grossProfitMargin != null ? formatPercent(result.ratios.grossProfitMargin) : null} />
              <FieldRow label="Net Profit Margin" value={result.ratios.netProfitMargin != null ? formatPercent(result.ratios.netProfitMargin) : null} />
              <FieldRow label="Operating Profit Margin" value={result.ratios.operatingProfitMargin != null ? formatPercent(result.ratios.operatingProfitMargin) : null} />
              <FieldRow label="P/E Ratio" value={result.ratios.peRatio != null ? formatNumber(result.ratios.peRatio, { maximumFractionDigits: 2 }) : null} />
              <FieldRow label="Debt to Equity" value={result.ratios.debtToEquity != null ? formatNumber(result.ratios.debtToEquity, { maximumFractionDigits: 2 }) : null} />
              <FieldRow label="Free Cash Flow Ratio" value={result.ratios.freeCashFlowRatio != null ? formatNumber(result.ratios.freeCashFlowRatio, { maximumFractionDigits: 3 }) : null} />
              <FieldRow label="Operating Cash Flow" value={result.ratios.operatingCashFlow != null ? formatNumber(result.ratios.operatingCashFlow, { notation: "compact", maximumFractionDigits: 2 }) : null} />
            </CardContent>
          </Card>

          <Card className="rounded-2xl transition hover:shadow-lg hover:shadow-[var(--shadow-glow)]">
            <CardHeader>
              <CardTitle>Technical</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">SMA 50</p>
                  {result.techIndicators.sma50 && result.techIndicators.sma50.length >= 2 ? (
                    <Sparkline data={result.techIndicators.sma50} height={28} className="text-[hsl(var(--brand-cyan))]" />
                  ) : result.techIndicators.sma50 && result.techIndicators.sma50.length === 1 ? (
                    <span className="text-sm">{formatNumber(result.techIndicators.sma50[0], { maximumFractionDigits: 2 })}</span>
                  ) : (
                    <span className="text-xs text-muted-foreground">Not present in document</span>
                  )}
                </div>
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">SMA 200</p>
                  {result.techIndicators.sma200 && result.techIndicators.sma200.length >= 2 ? (
                    <Sparkline data={result.techIndicators.sma200} height={28} className="text-[hsl(var(--brand-teal))]" />
                  ) : result.techIndicators.sma200 && result.techIndicators.sma200.length === 1 ? (
                    <span className="text-sm">{formatNumber(result.techIndicators.sma200[0], { maximumFractionDigits: 2 })}</span>
                  ) : (
                    <span className="text-xs text-muted-foreground">Not present in document</span>
                  )}
                </div>
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">MACD</p>
                  {result.techIndicators.macd && result.techIndicators.macd.length >= 2 ? (
                    <Sparkline data={result.techIndicators.macd} height={28} className="text-[hsl(var(--brand-green))]" />
                  ) : result.techIndicators.macd && result.techIndicators.macd.length === 1 ? (
                    <span className="text-sm">{formatNumber(result.techIndicators.macd[0], { maximumFractionDigits: 2 })}</span>
                  ) : (
                    <span className="text-xs text-muted-foreground">Not present in document</span>
                  )}
                </div>
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">RSI</p>
                  {result.techIndicators.rsi && result.techIndicators.rsi.length >= 2 ? (
                    <Sparkline data={result.techIndicators.rsi} height={28} className="text-primary" />
                  ) : result.techIndicators.rsi && result.techIndicators.rsi.length === 1 ? (
                    <span className="text-sm">{formatNumber(result.techIndicators.rsi[0], { maximumFractionDigits: 2 })}</span>
                  ) : (
                    <span className="text-xs text-muted-foreground">Not present in document</span>
                  )}
                </div>
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">VIX</p>
                  {result.techIndicators.vix && result.techIndicators.vix.length >= 2 ? (
                    <Sparkline data={result.techIndicators.vix} height={28} className="text-destructive" />
                  ) : result.techIndicators.vix && result.techIndicators.vix.length === 1 ? (
                    <span className="text-sm">{formatNumber(result.techIndicators.vix[0], { maximumFractionDigits: 2 })}</span>
                  ) : (
                    <span className="text-xs text-muted-foreground">Not present in document</span>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-2xl transition hover:shadow-lg hover:shadow-[var(--shadow-glow)]">
            <CardHeader>
              <CardTitle>Macro</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <FieldRow label="GDP" value={result.macro.gdp != null ? formatNumber(result.macro.gdp, { maximumFractionDigits: 2 }) : null} />
              <FieldRow label="Interest Rates" value={result.macro.interestRates != null ? formatNumber(result.macro.interestRates, { maximumFractionDigits: 2 }) : null} />
              <FieldRow label="CPI" value={result.macro.cpi != null ? formatNumber(result.macro.cpi, { maximumFractionDigits: 2 }) : null} />
            </CardContent>
          </Card>

          <Card className="rounded-2xl transition hover:shadow-lg hover:shadow-[var(--shadow-glow)]">
            <CardHeader>
              <CardTitle>Industry</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <FieldRow label="GICS" value={result.industry.gics ?? null} />
              <FieldRow label="Sector" value={result.industry.sector ?? null} />
              <FieldRow label="Embedding" value={result.industry.embedding?.length ? `${result.industry.embedding.length} dims` : null} />
            </CardContent>
          </Card>
        </div>
      </section>
    </main>
  );
};

export default Results;

import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { Extracted } from "@/types/extracted";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import Sparkline from "@/components/Sparkline";
import { Button } from "@/components/ui/button";

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
    document.title = result?.stockName
      ? `${result.stockName} Analysis — Results`
      : "Document Analysis Results";
  }, [result?.stockName]);

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

  const decisionVariant = result.decision === "BUY" ? "default" : "destructive" as const;

  return (
    <main>
      <section className="container mx-auto px-6 py-10">
        <header className="mx-auto mb-8 max-w-4xl text-center">
          <h1 className="mb-3 text-4xl font-bold tracking-tight">Analysis Results</h1>
          <p className="text-muted-foreground">Document type and detected stock name are shown below with overall decision.</p>
        </header>

        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-3">
          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle>Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <p className="text-sm text-muted-foreground">Document Type</p>
                  <p className="text-lg font-medium">{result.documentType ?? "Not present in document"}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Detected Stock</p>
                  <p className="text-lg font-medium">{result.stockName ?? "Not present in document"}</p>
                </div>
                <div className="flex items-center gap-3">
                  <p className="text-sm text-muted-foreground">Decision</p>
                  <Badge variant={decisionVariant} className="text-base px-3 py-1">
                    {result.decision}
                  </Badge>
                </div>
                <div className="flex items-center gap-3">
                  <p className="text-sm text-muted-foreground">Confidence Grade</p>
                  <Badge variant="secondary" className="text-base px-3 py-1">{result.confidenceGrade}</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Cool highlights</CardTitle>
            </CardHeader>
            <CardContent>
              {result.highlights?.length ? (
                <ul className="list-disc pl-5 space-y-2">
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
          <Card>
            <CardHeader>
              <CardTitle>Ratios</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    <TableHead>Value</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell>Gross Profit Margin</TableCell>
                    <TableCell>{formatPercent(result.ratios.grossProfitMargin ?? null)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Net Profit Margin</TableCell>
                    <TableCell>{formatPercent(result.ratios.netProfitMargin ?? null)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Operating Profit Margin</TableCell>
                    <TableCell>{formatPercent(result.ratios.operatingProfitMargin ?? null)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>P/E Ratio</TableCell>
                    <TableCell>{formatNumber(result.ratios.peRatio, { maximumFractionDigits: 2 })}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Debt to Equity</TableCell>
                    <TableCell>{formatNumber(result.ratios.debtToEquity, { maximumFractionDigits: 2 })}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Free Cash Flow Ratio</TableCell>
                    <TableCell>{formatNumber(result.ratios.freeCashFlowRatio, { maximumFractionDigits: 3 })}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Operating Cash Flow</TableCell>
                    <TableCell>{formatNumber(result.ratios.operatingCashFlow, { notation: "compact", maximumFractionDigits: 2 })}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Tech Indicators</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">SMA 50</p>
                  <Sparkline data={result.techIndicators.sma50 ?? null} className="text-brand-cyan" />
                </div>
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">SMA 200</p>
                  <Sparkline data={result.techIndicators.sma200 ?? null} className="text-brand-teal" />
                </div>
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">MACD</p>
                  <Sparkline data={result.techIndicators.macd ?? null} className="text-brand-green" />
                </div>
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">RSI</p>
                  <Sparkline data={result.techIndicators.rsi ?? null} className="text-primary" />
                </div>
                <div>
                  <p className="mb-2 text-sm text-muted-foreground">VIX</p>
                  <Sparkline data={result.techIndicators.vix ?? null} className="text-destructive" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Macro</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableBody>
                  <TableRow>
                    <TableCell>GDP</TableCell>
                    <TableCell>{formatNumber(result.macro.gdp, { maximumFractionDigits: 2 })}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Interest Rates</TableCell>
                    <TableCell>{formatNumber(result.macro.interestRates, { maximumFractionDigits: 2 })}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>CPI</TableCell>
                    <TableCell>{formatNumber(result.macro.cpi, { maximumFractionDigits: 2 })}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Industry</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <p className="text-sm text-muted-foreground">GICS</p>
                  <p className="font-medium">{result.industry.gics ?? "Not present in document"}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Sector</p>
                  <p className="font-medium">{result.industry.sector ?? "Not present in document"}</p>
                </div>
                <div className="sm:col-span-2">
                  <p className="text-sm text-muted-foreground">Embedding</p>
                  <p className="font-medium">
                    {result.industry.embedding?.length ? `${result.industry.embedding.length} dims` : "Not present in document"}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </main>
  );
};

export default Results;
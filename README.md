# Stake.ai

Turn financial documents into actionable signals. Upload a PDF, pick a strategy, and get a transparent BUY or SELL with a letter grade, plus highlights, ratios, sentiment, and risk flags.

## What this app does

- **Document intake**: Upload earnings reports, 10-Q/10-K, press releases, or investor letters. Text-only parsing, no OCR.
- **Smart extraction**: Classifies document type and ticker if present. Detects key fields like revenue, margins, leverage, cash flows. Fields that are missing show as “Not present.”
- **Insights**: Highlights notable sentences, tags risk phrases, and computes finance-focused sentiment.
- **Anomaly checks**: Compares extracted values against a FinanceBench-style baseline to flag large deviations.
- **Decision engine**: Combines sentiment, anomaly count, P/E availability, and leverage into a deterministic BUY or SELL with A-D confidence.
- **Signals (optional Stage 2)**: Computes SMA 50/200, MACD, RSI, moving averages, and basic OHLCV stats when you provide market data.
- **Clear UI**: Next.js 14 + TypeScript + Tailwind. Fields render only when present.

> Not financial advice. For research and demo use.

## Why it stands out

- **Document-grounded first**: Decisions trace back to facts found in the file.
- **Transparent logic**: Simple rules you can audit, not a black box.
- **Graceful nulls**: The UI shows “Not present” instead of guessing.
- **Modular**: Plug in new extractors or signals without changing the UI contract.

## Architecture

- **Frontend**: Next.js 14, TypeScript, Tailwind, shadcn/ui.
- **Server**: Next.js API routes for parsing and scoring.
- **Parsers**: Text-only PDF parsing with finance-tuned heuristics.
- **Anomaly store**: Lightweight baseline cache for common ratios.
- **Decision engine**: Rule set that takes sentiment, anomalies, leverage, and P/E to produce a badge and grade.
- **Model wiring**: Extractor, sentiment, anomaly checker, and decision engine communicate through one JSON schema. Adapters keep interfaces stable.

### Model buckets by stock type

Large caps, ETFs, and high beta names behave differently. Stage 2 supports per-bucket calibration and routing:
- Shared features
- Separate calibration for each bucket
- Simple registry that maps tickers to a bucket, with fallbacks

### Prerequisites if you want to download and run the app
- Node 18+
- pnpm or npm
- Optional: an LLM key if you enable any LLM-backed features

### Acknowledgments
Lovable for rapid UI scaffolding

Mentors and organising team of MIT Global AI Hackathon 2025




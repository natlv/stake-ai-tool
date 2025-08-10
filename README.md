Stake.ai
Turn financial documents into actionable signals. Upload a PDF, pick a strategy, and get a transparent BUY or SELL with a letter grade, plus highlights, ratios, sentiment, and risk flags.

What this app does
Document intake: Upload earnings reports, 10-Q/10-K, press releases, or investor letters. Text-only parsing, no OCR.

Smart extraction: Classifies document type and ticker if present. Detects key fields like revenue, margins, leverage, cash flows. Fields that are missing show as “Not present.”

Insights: Highlights notable sentences, tags risk phrases, and computes finance-focused sentiment.

Anomaly checks: Compares extracted values against a FinanceBench-style baseline to flag large deviations.

Decision engine: Combines sentiment, anomaly count, P/E availability, and leverage into a deterministic BUY or SELL with A–D confidence.

Signals (optional Stage 2): Computes SMA 50/200, MACD, RSI, moving averages, and basic OHLCV stats when you provide market data.

Clear UI: Next.js 14 + TypeScript + Tailwind. Fields render only when present.

Not financial advice. For research and demo use.

Why it stands out
Document-grounded first: Decisions trace back to facts found in the file.

Transparent logic: Simple rules you can audit, not a black box.

Graceful nulls: The UI shows “Not present” instead of guessing.

Modular: Plug in new extractors or signals without changing the UI contract.

Architecture
Frontend: Next.js 14, TypeScript, Tailwind, shadcn/ui.

Server: Next.js API routes for parsing and scoring.

Parsers: Text-only PDF parsing with finance-tuned heuristics.

Anomaly store: Lightweight baseline cache for common ratios.

Decision engine: Rule set that takes sentiment, anomalies, leverage, and P/E to produce a badge and grade.

Model wiring: Extractor, sentiment, anomaly checker, and decision engine communicate through one JSON schema. Adapters keep interfaces stable.

Model buckets by stock type
Large caps, ETFs, and high beta names behave differently. Stage 2 supports per-bucket calibration and routing:

Shared features

Separate calibration for each bucket

Simple registry that maps tickers to a bucket, with fallbacks

Getting started
Prerequisites
Node 18+

pnpm or npm

Optional: an LLM key if you enable any LLM-backed features

Install
bash
Copy
Edit
# clone
git clone <your-repo-url>
cd <your-repo-folder>

# install
pnpm install
# or: npm install
Run dev
bash
Copy
Edit
pnpm dev
# or: npm run dev
App runs on http://localhost:3000

Build and start
bash
Copy
Edit
pnpm build && pnpm start
# or: npm run build && npm start
Environment variables
Create .env.local and set what you need:

ini
Copy
Edit
# Optional if using any LLM helpers
OPENAI_API_KEY=sk-...
# PDF size limits and upload dir
MAX_UPLOAD_MB=25
UPLOAD_DIR=.uploads
# Toggle stage 2 signals when you have OHLCV
ENABLE_STAGE2=true
Usage
Open the app and upload a PDF.

Choose an investment strategy profile.

Review the output panel: document type, ticker, highlights, ratios, sentiment, risk flags, anomalies, and BUY or SELL with confidence.

If Stage 2 is enabled and you have market data, open the Signals tab to view SMA 50/200, MACD, RSI, and basic trend filters.

Data flow
Parse → extract text from PDF.

Detect → classify doc type and ticker.

Extract → pull numeric fields and key phrases.

Score → compute sentiment and anomaly flags.

Decide → deterministic engine returns BUY or SELL with grade.

Render → UI only shows fields that exist.

API snapshot
POST /api/parse
Body: { file }
Returns: { text, metadata }

POST /api/extract
Body: { text }
Returns: { fields, missing }

POST /api/score
Body: { fields }
Returns: { sentiment, anomalies }

POST /api/decide
Body: { sentiment, anomalies, fields }
Returns: { decision: "BUY" | "SELL", grade: "A" | "B" | "C" | "D" }

POST /api/signals (optional Stage 2)
Body: { ohlcv }
Returns: { sma50, sma200, macd, rsi, crossovers }

Development notes
This codebase was bootstrapped with Lovable. It worked well once we changed our approach:

A single mega prompt caused a 15 minute stall.

Splitting work into small prompts per page or feature improved speed and quality.

Keep a short design spec in the repo and feed that into each prompt to stay consistent.

Troubleshooting
Mic or screen permissions on macOS: System Settings → Privacy & Security → allow Screen Recording or Microphone for your browser if you test any capture features.

PDF has images only: The parser is text-only. Use a text-based PDF or run OCR before upload.

Signals are empty: Enable Stage 2 and provide OHLCV data.

Roadmap
OCR fallback for image-only PDFs

Better anomaly baselines per sector and regime

Strategy backtests with walk-forward splits

Export to PDF and shareable links

Pluggable LLM summaries with strict grounding

Contributing
PRs and issues are welcome. Please keep changes small and focused.

License
MIT

Acknowledgments
Lovable for rapid UI scaffolding

Mentors and organising team of MIT Global AI Hackathon 2025








Ask ChatGPT

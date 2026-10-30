# Aegis

Research foundation for an autonomous quant stack. This repository is phases 1 through 3 of a seven-phase platform: a Massive (formerly Polygon.io) 1-minute bar pipeline, a NautilusTrader EMA-crossover backtest with a deterministic risk engine, and an OpenAI research loop that can propose and implement signal plug-ins. No model is allowed to override the risk engine.

Massive supplies the bars. NautilusTrader runs the simulation. Alpaca is reserved for paper trading in a later phase. The phase-3 agents research and write signal modules. They do not size risk, and this phase does not start a backtest.

## Phase roadmap

| Phase | What | Status |
| --- | --- | --- |
| 1 | Massive 1-minute pipeline, Parquet cache, validation, point-in-time universe | This build |
| 2 | NautilusTrader EMA crossover, costs, risk engine, walk-forward, holdout | This build |
| 3 | Orchestrator, Researcher, Implementer, Naysayer | This build |
| 4 | Backtesting Analyst, ML Specialist (XGBoost), Risk Analyst | Not started |
| 5 | Experiment loop, MLflow, Postgres | Not started |
| 6 | Alpaca paper trading | Not started |
| 7 | Independent validation, monitoring, human-approved live trading | Not started |

`aegis.agents` is the phase-3 loop. `aegis.ml`, `aegis.execution`, and `aegis.experiments` are still reserved. The strategy harness (`aegis.strategies.harness`) is the safety-critical runner for plug-ins; until that module lands, agents code against the stub in `aegis.agents.signal_contract`.

## Setup

Python 3.11 or 3.12. This build is tested on 3.12 with NautilusTrader 1.231.0.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put your key in `.env` or the environment. Config is read from the environment only. Nothing in the repo contains a secret.

```bash
MASSIVE_API_KEY=your-key
TRADING_MODE=paper
LIVE_TRADING_ENABLED=false
```

`TRADING_MODE=live` or `LIVE_TRADING_ENABLED=true` is rejected. This build has no live-order path.

You can run the synthetic backtest and the test suite with no keys:

```bash
aegis backtest demo
pytest
```

`aegis backtest demo` replays two synthetic regular sessions (3–4 January 2024) through the same EMA strategy, cost model, and risk engine as a real run. The path is built so a cross and a 2.5R target both happen. The Sharpe on two daily returns is a smoke check, not a performance claim.

## Free-tier Massive access

The default plan is the free tier:

- **5 requests per minute.** `MASSIVE_RATE_LIMIT_PER_MINUTE` raises this for a paid plan. Leaving it above 5 while `MASSIVE_PLAN=free` just produces 429s; the client backs off.
- **About two years of history.** `MASSIVE_MAX_LOOKBACK_DAYS=730`. Older start dates are clamped and the command says so.
- **Delayed, end-of-day data.** The free plan never requests today's Eastern session. The latest bar is the previous completed NYSE session. Set `MASSIVE_PLAN=paid` to include today's session after 16:15 America/New_York. There is no websocket or snapshot client in this build.

Each aggregates call uses `limit=50000` and `adjusted=false`. `limit` is a scan budget of base minutes, not a page size. Extended hours are about 960 minutes a session, so a chunk of **70 calendar days** scans about 46,000 base minutes and usually comes back in one response. The client still follows `next_url` if Massive paginates. 429s and 5xx responses retry with exponential backoff and honor `Retry-After`. The api key is sent as a bearer token and is redacted if it ever appears in a URL.

### How long a full backfill takes

For the default 20-name universe, as of 30 October 2026, a cold cache covering the whole free-tier window (2024-10-29 through 2026-10-29) is:

| Work | Requests |
| --- | --- |
| 1-minute bars, 11 chunks per symbol | 220 |
| Splits and dividends, once per symbol | 40 |
| **Total** | **260** |

At 5 requests per minute that is **52 minutes**. `aegis data plan` prints the same estimate for whatever window and cache you actually have, including days already on disk.

A later daily update is one bar request per symbol that is missing data, plus two corporate-action requests per symbol if you refresh splits and dividends (about 12 minutes for all 20). `aegis data update --skip-corporate-actions` skips that refresh (about 4 minutes) when you know there was no split.

### Resuming

Session partitions are the checkpoint: `data/cache/bars/symbol=AAPL/date=2024-01-03/bars.parquet`. A chunk is written only after its bars are in memory, via a temp file and rename. Stop the process and run the same command again. Finished days are not downloaded again. The worst repeat is the single chunk that was in flight. `data/cache/manifest.jsonl` is a log of completed chunks. Resume does not trust it; it trusts the parquet files.

Corporate actions are checkpointed separately and skipped once they already cover the window end.

The cache stores **raw** prints. Split adjustment (and optional dividend adjustment) happens when bars are read. Splits after the read's `as_of` date are ignored, so a backtest does not use a split that had not happened yet.

## Commands

```bash
# Request count and wall time. Does not call Massive.
aegis data plan

# First pull. Clamped to the free-tier window. Safe to re-run.
aegis data backfill --start 2024-10-29

# Same, for two names, through an explicit end date.
aegis data backfill --start 2025-01-02 --end 2025-06-30 --symbols AAPL,MSFT

# Fill holes and extend whatever is already cached.
aegis data update
aegis data update --skip-corporate-actions

# Missing minutes, inverted bars, extreme returns, extended-hours prints.
aegis data validate --start 2025-01-02 --end 2025-06-30

# Research backtest on the cache. Refuses the holdout.
aegis backtest run --start 2025-01-02 --end 2026-06-30

# Walk-forward fold dates. Also refuses the holdout. Does not run a backtest.
aegis backtest walk-forward --start 2025-01-02 --end 2026-06-30 --train-days 90 --test-days 30

aegis backtest demo
aegis backtest demo --json

# Phase 3. Needs OPENAI_API_KEY. Does not backtest.
aegis research run --objective "Mean reversion after a morning spike." --max-iterations 2
```

`backtest run` uses a 20/50 EMA and a 14-period ATR. `backtest demo` uses 5/12 and a 10-period ATR so the short fixture can warm up. Both go through the same risk engine.

## Universe

`config/universe.yaml` is twenty liquid names (AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, TSLA, AMD, COST, JPM, V, UNH, XOM, LLY, JNJ, MA, HD, WMT, BAC). That is small enough for a free-tier backfill of the full two-year window to finish in about an hour.

Membership has `start` and `end`. `end: null` means still listed. `select_as_of` drops a name outside that window, which is how delisted symbols will be included later without pretending they traded forever. The starter list is today's survivors, not a historical index. Survivorship bias is real until delisted names are added. A commented Lehman-style example in the YAML shows the shape.

Listing MIC is metadata. Every research instrument is simulated on one Nautilus venue, `XNAS`, so the account and the five-position cap are portfolio-wide.

## Risk engine

`aegis.backtest.risk.RiskEngine` is plain Python. The strategy asks it before every entry and trades only an approved quantity, stop, and target. The caps are the spec, not environment variables:

- 1% of equity at risk per trade
- 5 simultaneous positions
- 25% of equity per position
- 5% daily loss cutoff (flattens and blocks new entries for that session; it does not latch the kill switch)
- ATR stop, 2.0 times Wilder ATR, rounded to cents
- Initial target at least 2.5R, rounded away from the entry so the ratio cannot fall under 2.5
- Entries only from 9:35 to 15:30 America/New_York, weekdays
- Flatten from 15:55 America/New_York
- `RISK_KILL_SWITCH=true` latches an emergency stop until something in-process calls `reset`

If the runner does not install the engine, the strategy refuses to trade. The baseline is long-only. The engine also knows how to price a short, for later strategies.

Costs, on the venue rather than inside the signal: **$0.005 per share** plus **5 bps of notional on every fill**, both sides. That slippage is deterministic. It is not Nautilus's one-tick coin flip. The account is margin with 2x leverage so five 25% positions are possible; the risk engine is what actually caps exposure.

The report prints total return, Sharpe (daily returns, `sqrt(252)`, zero risk-free rate), max drawdown, win rate, average R-multiple, trade count, and equal-weight buy-and-hold of the symbols in the run. R is gross price R against the initial stop. PnL is net of commission and slippage. Buy-and-hold is close-to-close, fully invested, without costs, so it is not the same capital schedule as the intraday strategy.

## Holdout

`HOLDOUT_START=2026-07-01` and `HOLDOUT_END=2026-10-31` are the protected final window. Backfill is allowed to **store** those dates so they exist later. Research reads are not allowed to **open** them.

`aegis backtest run`, `aegis backtest walk-forward`, and `BarCache.read_bars` (the default purpose is `research`) raise if the requested range overlaps the holdout. There is no research function that returns the overlapping part and drops the rest. A full free-tier backfill includes this window in the cache; point research backtests at a range that ends on or before 2026-06-30. Change the holdout dates only on purpose. This build does not include the one-shot final evaluation that will be allowed to read it.

## Research agents

Phase 3 runs four roles: Orchestrator, Researcher, Algorithm Implementer, and Naysayer. The runner is a thin tool loop in the shape of the OpenAI Agents SDK (instructions, scoped function tools, a pydantic `output_type`, a turn limit). It calls the OpenAI Chat Completions API. Tests never do. They inject a scripted client.

```bash
export OPENAI_API_KEY=sk-...
# Optional. Default is gpt-6.1-sol. Prices below match that model ($2 / $10 per million).
export AEGIS_RESEARCH_MODEL=gpt-6.1-sol
# Optional per-role overrides. Empty falls back to AEGIS_RESEARCH_MODEL.
# export AEGIS_IMPLEMENTER_MODEL=gpt-6-luna
aegis research run --objective "Short lookback momentum on the liquid universe." --max-iterations 2
```

`--max-iterations` is the propose/critique rounds. The implementer then writes one candidate. The command prints a stop reason and the audit path. `advance_to_backtest: true` means the gates passed. `backtest_started` stays false. Phase 4 is what will run the simulation.

What the run actually does:

1. The orchestrator restates the objective. Constraints that ask to loosen risk, skip the naysayer, or read the holdout are dropped.
2. The researcher proposes entry rules, exit rules, market assumptions, and a falsifiable hypothesis. It can list the universe and ask whether a date window is outside the holdout. It cannot read bars.
3. The naysayer returns a critique with findings, severity, evidence, and a pass/fail verdict. An unresolved critical finding forces `fail`.
4. The implementer writes `candidate.py` (a `Params` model plus one `SignalModel`) and `test_candidate.py`, only inside a scratch workspace. The workflow runs that source in the sandbox again and ignores a claim that the tests passed.

A candidate is kept only as a versioned directory: `data/research/<run_id>/<sha256>/` with the signal module, the tests, `proposal.json`, `critique.json`, and `manifest.json`. The hash is the signal source. The same directory is written when the sandbox rejects the file, so the audit can point at it. Nothing is copied into `src/aegis/strategies/`.

The plug-in returns an action (`enter_long`, `enter_short`, `exit`, `hold`) and a reason. Quantity, stop, and target stay on `RiskEngine` with the default `RiskLimits`. The sandbox probes that engine on a fixed 10:00 Eastern entry and records the quantity. A plug-in that imports an order API, constructs `RiskLimits`, reads `aegis.data`, or mentions a holdout date is rejected before it runs. The child process gets a scrubbed environment, a CPU and address-space limit, a wall-clock timeout, no network, and no write outside the scratch directory. `aegis/strategies/harness.py`, `aegis/backtest/risk.py`, `aegis/data/`, and `aegis/holdout.py` are not writable.

The orchestrator can ask to override a risk limit or a failed validation. `decide_advancement` records the request and does not honor it. Critical findings and a failed sandbox both stop the run before any backtest.

Every model message and tool call is appended to `data/audit/<run_id>.jsonl`. Prompts live in `src/aegis/agents/prompts/<role>/v1.md` (`AEGIS_PROMPT_VERSION` selects the file). The OpenAI key is read inside the client constructor. It is not a tool argument and it is scrubbed from the audit if it shows up in text. Tool handlers do not receive the settings object.

Budgets are hard stops: `AEGIS_MAX_INPUT_TOKENS` (150000), `AEGIS_MAX_OUTPUT_TOKENS` (40000), `AEGIS_MAX_COST_USD` (3), `AEGIS_MAX_LLM_CALLS` (24). Crossing one aborts the run before the next call. The default prices are `AEGIS_PRICE_INPUT_PER_MTOK=2` and `AEGIS_PRICE_OUTPUT_PER_MTOK=10`. Change those if you change the model.

`pytest` covers this loop with a scripted model, including one full cycle that executes the real sandbox. No test calls OpenAI.

## Tests

```bash
pytest
```

The suite covers the HTTP client with a fake transport (pagination, `limit=50000`, 429 retry, secret redaction), the rate limiter, cache partitions, resume-without-redownload, split and dividend adjustment, bar validation, holdout refusal, every risk rule, walk-forward folds, an end-to-end Nautilus backtest on the synthetic fixture, and the phase-3 agent loop with a scripted model (sandbox policy, budgets, secret redaction, and one full research cycle). No test calls Massive, Alpaca, or OpenAI.

## NautilusTrader

Pinned to **1.231.0** on Python 3.12. Current Nautilus 2.x docs describe a pre-release (`2.0.0rc`) with a different import layout and Python 3.13+. This code follows the 1.231 API (`BacktestEngine`, `StrategyConfig`, `Equity`, `Bar`).

Other behavior that matters when you read a report:

- Bars are built directly into Nautilus `Bar` objects. The 1.231 `BarDataWrangler` takes a NumPy view of a pandas frame, and pandas 3 can hand that view back read-only.
- A market order submitted from `on_bar` fills on the next bar. The 5 bps charge is the adverse-cost buffer for that.
- `bar_adaptive_high_low_ordering` is on, and stop orders are not rejected when the trigger is already in the market, so a gap through the stop flattens instead of leaving a naked position.
- The engine logs at `ERROR` (`NAUTILUS_LOG_LEVEL`) so the report is readable. Nautilus 1.231 emits a pandas `Timestamp.utcnow` deprecation while it runs; it does not change the fills.

## Layout

```
src/aegis/
  config.py            environment settings
  cli.py               aegis data|backtest|research
  holdout.py           protected window
  data/                Massive client, cache, query, validation, universe
  backtest/            risk engine, EMA strategy, report, walk-forward
  agents/              phase-3 prompts, tools, sandbox, workflow
  ml/  execution/  experiments/             later phases
config/universe.yaml
```

`agents/signal_contract.py` is the plug-in seam (`SignalModel`, `Signal`, `BarView`, the AST policy, the registry). When `aegis.strategies.harness` defines the same names, the sandbox uses that policy and accepts that base class too. Generated modules stay in the research store. They do not edit the harness.

Cache files under `data/cache/` are gitignored.

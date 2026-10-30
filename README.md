# Aegis

Research foundation for an autonomous quant stack. Massive (formerly Polygon.io) supplies 1-minute bars. NautilusTrader runs the simulation. A deterministic risk engine sizes every order. Strategy plug-ins, an XGBoost signal ranker, and advisory portfolio analytics are the tools the later analyst agents will call. Those agents are not in this build. They will not size risk.

## Phase roadmap

| Phase | What | Status |
| --- | --- | --- |
| 1 | Massive 1-minute pipeline, Parquet cache, validation, point-in-time universe | This build |
| 2 | NautilusTrader backtest, costs, risk engine, walk-forward optimization, holdout | This build |
| 3 | Orchestrator, Researcher, Implementer, Naysayer | Not started |
| 4 | Strategy plug-ins, ML ranker, portfolio analytics, analyst report | Tooling in this build; the agents are not |
| 5 | Experiment loop and Postgres | Not started |
| 6 | Alpaca paper trading | Not started |
| 7 | Independent validation, monitoring, human-approved live trading | Not started |

`aegis.agents`, `aegis.execution`, and `aegis.experiments` are still empty. `aegis.ml` and `aegis.portfolio` are the Phase 4 tool layer.

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

# Research backtest on the cache. Refuses the holdout. Default strategy is ema_cross.
aegis backtest run --start 2025-01-02 --end 2026-06-30
aegis backtest run --strategy macd --timeframe 15min --start 2025-01-02 --end 2026-06-30

# Names the optimizer can search.
aegis optimize strategies

# Walk-forward fold dates. Also refuses the holdout. Does not run a backtest.
aegis backtest walk-forward --start 2025-01-02 --end 2026-06-30 --train-days 90 --test-days 30

aegis backtest demo
aegis backtest demo --json
```

## Walk-forward optimization

`aegis optimize` searches a registered strategy. Parameters and their default grids live on that strategy's Pydantic model (`SearchMeta` on each field), not in the optimizer. `ema_cross` sweeps fast period, slow period, ATR stop multiple, and bar timeframe. Bar timeframe aggregates the cached 1-minute bars (5min, 15min, 30min, 60min) before the backtest, aligned to 9:30 ET. ATR length stays at 14 unless you override it. Risk limits other than the ATR multiple stay fixed.

On 1-minute bars, EMA 20/50 trades about 1,200 times per name over 18 months and loses roughly 25–29% net on every name, with average net R near zero. Costs dominate that timeframe. Sweep `bar_timeframe` (and slower EMAs) instead of trusting the 1-minute default.

```bash
aegis optimize strategies
aegis optimize space ema_cross
```

For each fold the search runs every parameter set on the in-sample window, keeps the best in-sample Sharpe (ties keep the first set in the sorted grid), and runs that set once on the following out-of-sample window. Parameter sets that were not selected are not scored out of sample.

Rolling folds slide a fixed training window. Anchored folds keep the training start date and lengthen the window. `step-days` defaults to `test-days` and must be at least `test-days`, so out-of-sample windows do not overlap. The stitched curve compounds those out-of-sample segments only.

Trials run in separate processes, default `min(cpu-1, 6)`. Every in-sample evaluation is one trial. The ledger is Parquet (`params_json` holds the full set) and is readable with DuckDB. Out-of-sample columns are null unless that row was selected. The report prints stitched OOS Sharpe, the best single-fold OOS Sharpe, the trial count, the deflated Sharpe, and parameter stability across folds. Average R in every report is net of commission and slippage.

The holdout guard applies. A range that overlaps it is refused before any trial starts.

```bash
# Default ema_cross grid, but only 5- and 15-minute bars.
aegis optimize run \
  --strategy ema_cross \
  --start 2025-01-02 --end 2026-06-30 \
  --set bar_timeframe=5min,15min \
  --mode rolling --train-days 60 --test-days 20 --embargo-days 5 \
  --workers 4 \
  --ledger data/cache/trials/trials.parquet

# Replace the fast/slow grid too. Other axes keep the model's default grid.
aegis optimize run \
  --strategy ema_cross \
  --start 2025-01-02 --end 2026-06-30 \
  --set fast_period=12,20 --set slow_period=50,80 \
  --set atr_multiplier=2,3 --set bar_timeframe=15min \
  --method random --samples 12 --seed 7 --mode anchored

# Synthetic week, no API key. Smoke grid unless you pass --set.
aegis optimize demo --workers 2 --ledger data/cache/trials/trials.parquet
```

`data/cache/trials/oos_equity.parquet` is the stitched out-of-sample equity, tagged with `study_id` and `fold_index`.

```bash
python -c "import duckdb; print(duckdb.sql(\"SELECT strategy, n_trials, count(*) FROM read_parquet('data/cache/trials/trials.parquet') GROUP BY 1, 2\"))"
```

## Strategy plug-ins

`HarnessStrategy` is the only Nautilus strategy that submits orders. It registers indicators, asks `RiskEngine` for size, stop, and target, and enforces the entry window, the 15:55 flatten, the daily loss cutoff, and the kill switch. A plug-in is a `SignalModel`: on each completed bar it returns `enter_long`, `enter_short`, `exit`, or `none`, plus optional stop and target hints. A hint that would widen risk is ignored. A tighter stop is kept and the quantity stays the one the engine approved, so risk only shrinks.

Each plug-in is a `StrategyParams` subclass with `SearchMeta` on the tunable fields, including `bar_timeframe` (`1min`, `5min`, `15min`, `30min`, `60min`). `aegis optimize` reads that model. The cached bars stay 1-minute; `resample_bars` aggregates them to the session open before the backtest. New strategies default to 5-minute bars because a 1-minute EMA cross is cost-dominated. `ema_cross` stays on 1-minute bars so the synthetic baseline trade is unchanged.

Registered names:

| Name | Signal |
| --- | --- |
| `ema_cross` | Fast EMA crosses above the slow EMA. Report label remains `ema_crossover`. |
| `macd` | MACD line crosses above its signal line. |
| `opening_range_breakout` | Close breaks the opening-range high, once per session. |
| `mean_reversion` | Close is a negative Bollinger z-score and below session VWAP. |
| `volatility_breakout` | Close breaks the prior N-bar high. |

`aegis backtest demo --strategy macd` uses a short synthetic path and does not need a cache. `aegis backtest signals` runs the static policy check.

Signal modules may import the parameter model, the signal types, the registry helper, and `resample`. They may not import Nautilus, the broker, or call `submit_order`. `check_signal_source` is the check a later sandbox can run on generated text before import.

```python
from aegis.strategies.policy import check_signal_package
check_signal_package()
```

## ML signal ranker

`aegis.ml.train_signal_ranker` builds causal price, volume, and volatility features, labels candidate entry times with the net R of a long trade after commission and slippage, and fits XGBoost to predict that R. Validation is a purged, embargoed walk-forward: a training row whose label interval overlaps the test window is dropped. The search logs each trial to a local MLflow file store (`mlruns/` by default, gitignored). The result also includes a leakage report, a feature-importance stability report, and a comparison against taking every signal and against ranking by `ret_5`.

Any bar whose Eastern session sits in the holdout raises before a model is fit.

```python
from aegis.ml import train_signal_ranker
ranker, result = train_signal_ranker(bars, holdout=settings.holdout, tracking_uri="mlruns")
ranked = ranker.rank(features)
```

## Portfolio analytics

`aegis.portfolio.analyze_portfolio` is advisory. It reports return correlations, concentration (Herfindahl and the 25% cap as a warning), drawdown span, volatility and trend regime buckets, and four stress paths (scaled returns, correlation to 1, drop the largest weight, worst overlapping day). The allocation suggestion is inverse-volatility, long only, capped at 25% when that cap can be met. It does not call `RiskEngine.evaluate_entry` and it does not change `RiskLimits`.

```python
from aegis.portfolio import analyze_portfolio, series_from_report
report = analyze_portfolio([series_from_report(ema_report), series_from_report(macd_report)])
```

A new strategy is a module under `src/aegis/strategies/` with a `StrategyParams` subclass, a `SignalModel`, and a `register_signal(...)` call. Import it from `aegis.strategies` so the name shows up in `aegis optimize strategies`.

`backtest run` defaults to `ema_cross` (20/50 EMA, 14-period ATR, 1-minute bars). `backtest demo` uses 5/12 and a 10-period ATR so the short fixture can warm up. Both go through the same harness and the same risk engine.

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

The report prints total return, Sharpe (daily returns, `sqrt(252)`, zero risk-free rate), max drawdown, win rate, average R-multiple, trade count, and equal-weight buy-and-hold of the symbols in the run. R is net of commission and slippage: realized PnL divided by the initial risk in dollars. Buy-and-hold is close-to-close, fully invested, without costs, so it is not the same capital schedule as the intraday strategy.

`PerformanceReport.to_dict()` is the document the Backtesting Analyst will read (`schema_version` 1.1). It adds the parameter set, the cost assumptions, a snapshot of the risk limits that were installed, daily returns, profit factor, Sortino, and expectancy. `BacktestReport.from_performance` validates that JSON. The snapshot does not change the engine.

## Holdout

`HOLDOUT_START=2026-07-01` and `HOLDOUT_END=2026-10-31` are the protected final window. Backfill is allowed to **store** those dates so they exist later. Research reads are not allowed to **open** them.

`aegis backtest run`, `aegis backtest walk-forward`, and `BarCache.read_bars` (the default purpose is `research`) raise if the requested range overlaps the holdout. There is no research function that returns the overlapping part and drops the rest. A full free-tier backfill includes this window in the cache; point research backtests at a range that ends on or before 2026-06-30. Change the holdout dates only on purpose. This build does not include the one-shot final evaluation that will be allowed to read it.

## Tests

```bash
pytest
```

The suite covers the HTTP client with a fake transport (pagination, `limit=50000`, 429 retry, secret redaction), the rate limiter, cache partitions, resume-without-redownload, split and dividend adjustment, bar validation, holdout refusal, every risk rule, walk-forward folds, the optimizer ledger, an end-to-end Nautilus backtest on the synthetic fixture, the four signal plug-ins, the signal-module policy check, purged walk-forward training on synthetic bars, and the advisory portfolio report. No test calls Massive, Alpaca, or OpenAI.

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
  cli.py               aegis data|backtest|optimize
  holdout.py           protected window
  data/                Massive client, cache, query, validation, universe
  backtest/            risk engine, report, walk-forward, optimizer
  strategies/          harness, signal plug-ins, registry, resample, policy
  ml/                  features, net-R labels, XGBoost ranker, MLflow
  portfolio/           advisory correlation, stress, and allocation
  agents/  execution/  experiments/    later phases
config/universe.yaml
```

Cache files under `data/cache/` are gitignored.

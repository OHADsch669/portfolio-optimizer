"""
Walk-forward backtest: simulates running each strategy in real time.

On the first trading day of every month:
  1. Look back LOOKBACK_DAYS trading days (only data that existed at that moment)
  2. Estimate mu and Sigma from that window and compute the strategy's target weights
  3. Rebalance to the targets, paying transaction costs on what was traded
Between rebalances the portfolio is held: weights drift with prices.

Results (metrics + daily portfolio value) are saved to the backtests tables.
"""
import sqlite3

import numpy as np
import pandas as pd

import optimizer as opt
import stats

LOOKBACK_DAYS = 504        # ~2 years of trading days for estimation
MAX_WEIGHT = 0.20
COV_METHOD = "ledoit_wolf"
COST_BPS = 10              # transaction cost: 10 basis points = 0.10% of the amount traded
BENCHMARK = "SPY"

# Each strategy: (mu, sigma, rf) -> weights
STRATEGIES = {
    "equal_weight": lambda mu, sigma, rf: opt.equal_weight(len(mu)),
    "min_variance": lambda mu, sigma, rf: opt.min_variance(sigma, MAX_WEIGHT),
    "max_sharpe":   lambda mu, sigma, rf: opt.max_sharpe(mu, sigma, rf, MAX_WEIGHT),
    "risk_parity":  lambda mu, sigma, rf: opt.risk_parity(sigma),
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def rebalance_dates(index: pd.DatetimeIndex, lookback: int) -> pd.DatetimeIndex:
    """First trading day of each month, once enough history exists."""
    firsts = index.to_series().groupby(index.to_period("M")).first()
    return pd.DatetimeIndex(firsts[firsts >= index[lookback]])


def run_backtest(returns: pd.DataFrame, rf_daily: pd.Series, strategy) -> tuple[pd.Series, float]:
    """
    Returns (daily net portfolio returns, total turnover).
    Weights for day d use only returns strictly before d: no look-ahead.
    """
    dates = rebalance_dates(returns.index, LOOKBACK_DAYS)
    n = returns.shape[1]
    current_w = np.zeros(n)          # start in cash
    daily = []
    total_turnover = 0.0

    for i, d in enumerate(dates):
        window = returns.loc[:d].iloc[-LOOKBACK_DAYS - 1:-1]
        mu = stats.expected_returns(window).values
        sigma = stats.covariance_matrix(window, method=COV_METHOD).values
        rf = float(rf_daily.loc[window.index].mean() * stats.TRADING_DAYS)

        try:
            target = strategy(mu, sigma, rf)
        except RuntimeError:
            target = opt.min_variance(sigma, MAX_WEIGHT)   # fallback if the solver fails

        turnover = np.abs(target - current_w).sum()
        total_turnover += turnover

        # Hold `target` until the next rebalance date
        end = dates[i + 1] if i + 1 < len(dates) else None
        period = returns.loc[d:end].iloc[:-1] if end is not None else returns.loc[d:]
        growth = (1 + period).cumprod()                 # value of $1 in each stock
        value = growth.values @ target                  # portfolio value, starts near 1
        port_ret = pd.Series(value, index=period.index).pct_change()
        port_ret.iloc[0] = value[0] - 1
        port_ret.iloc[0] = (1 + port_ret.iloc[0]) * (1 - turnover * COST_BPS / 10_000) - 1
        daily.append(port_ret)

        drifted = target * growth.values[-1]
        current_w = drifted / drifted.sum()

    return pd.concat(daily), total_turnover


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(r: pd.Series, rf_daily: pd.Series) -> dict:
    equity = (1 + r).cumprod()
    years = len(r) / stats.TRADING_DAYS
    excess = r - rf_daily.reindex(r.index).fillna(0)
    return {
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "volatility": r.std() * np.sqrt(stats.TRADING_DAYS),
        "sharpe": excess.mean() / excess.std() * np.sqrt(stats.TRADING_DAYS),
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "final_value": equity.iloc[-1],
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_backtest(conn: sqlite3.Connection, name: str, r: pd.Series, m: dict) -> int:
    cur = conn.execute(
        """INSERT INTO backtests (strategy, start_date, end_date, lookback_days, rebalance_freq,
               transaction_cost_bps, max_weight, cov_method, cagr, volatility, sharpe, max_drawdown)
           VALUES (?, ?, ?, ?, 'monthly', ?, ?, ?, ?, ?, ?, ?)""",
        (name, r.index[0].strftime("%Y-%m-%d"), r.index[-1].strftime("%Y-%m-%d"), LOOKBACK_DAYS,
         COST_BPS, MAX_WEIGHT, COV_METHOD,
         m["cagr"], m["volatility"], m["sharpe"], m["max_drawdown"]),
    )
    backtest_id = cur.lastrowid
    equity = (1 + r).cumprod()
    conn.executemany(
        "INSERT INTO backtest_values (backtest_id, date, portfolio_value) VALUES (?, ?, ?)",
        [(backtest_id, d.strftime("%Y-%m-%d"), float(v)) for d, v in equity.items()],
    )
    return backtest_id


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def main() -> None:
    with sqlite3.connect(stats.DB_PATH) as conn:
        tickers = stats.get_stock_tickers(conn)
        returns = stats.daily_returns(stats.load_prices(conn, tickers))
        spy = stats.daily_returns(stats.load_prices(conn, [BENCHMARK]))[BENCHMARK]
        rf = pd.read_sql_query("SELECT date, rate FROM risk_free_rates", conn,
                               parse_dates=["date"], index_col="date")["rate"]
        rf_daily = (rf / 100 / stats.TRADING_DAYS).reindex(returns.index).ffill().fillna(0)

        results = {}
        for name, strategy in STRATEGIES.items():
            print(f"Running {name}...")
            r, turnover = run_backtest(returns, rf_daily, strategy)
            m = metrics(r, rf_daily)
            m["turnover_per_year"] = turnover / (len(r) / stats.TRADING_DAYS)
            save_backtest(conn, name, r, m)
            results[name] = (r, m)

        # Benchmark over exactly the same days, no trading costs
        first_day = next(iter(results.values()))[0].index[0]
        spy_r = spy.loc[first_day:]
        m = metrics(spy_r, rf_daily)
        m["turnover_per_year"] = 0.0
        save_backtest(conn, "benchmark_spy", spy_r, m)
        results["SPY (benchmark)"] = (spy_r, m)
        conn.commit()

    table = pd.DataFrame({name: m for name, (_, m) in results.items()}).T
    table = table.sort_values("sharpe", ascending=False)
    print(f"\nOut-of-sample backtest: {first_day.date()} to {spy_r.index[-1].date()}, "
          f"monthly rebalance, {LOOKBACK_DAYS}-day lookback, {COST_BPS} bps costs\n")
    print(table.to_string(formatters={
        "cagr": "{:.1%}".format, "volatility": "{:.1%}".format, "sharpe": "{:.2f}".format,
        "max_drawdown": "{:.1%}".format, "final_value": "{:.2f}x".format,
        "turnover_per_year": "{:.0%}".format,
    }))


if __name__ == "__main__":
    main()
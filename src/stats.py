"""
Statistics engine: turns stored prices into the two inputs every
portfolio optimizer needs:
    mu    - expected annual return per stock     (vector)
    sigma - annual covariance matrix of returns  (matrix)
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "portfolio.db"
TRADING_DAYS = 252  # trading days in a year, used to annualize daily numbers


def get_stock_tickers(conn: sqlite3.Connection) -> list[str]:
    """Tickers we optimize over (stocks only, without the SPY benchmark)."""
    rows = conn.execute("SELECT ticker FROM assets WHERE asset_type = 'stock'").fetchall()
    return [row[0] for row in rows]


def load_prices(
    conn: sqlite3.Connection,
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load adjusted prices as a wide table: one row per date, one column per ticker.
    (The DB stores long format; the math needs wide format.)
    """
    placeholders = ",".join("?" * len(tickers))
    query = f"SELECT date, ticker, adj_close FROM prices WHERE ticker IN ({placeholders})"
    params: list = list(tickers)
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    wide = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return wide[tickers]  # keep the requested column order


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage change: (P_today / P_yesterday) - 1."""
    return prices.pct_change().dropna()


def expected_returns(returns: pd.DataFrame) -> pd.Series:
    """Historical mean daily return, annualized."""
    return returns.mean() * TRADING_DAYS


def covariance_matrix(returns: pd.DataFrame, method: str = "sample") -> pd.DataFrame:
    """
    Annualized covariance matrix.
    - 'sample':      the plain historical covariance
    - 'ledoit_wolf': shrinks the sample covariance toward a simpler, more
                     stable target; less noisy, so optimized weights are less extreme
    """
    if method == "sample":
        cov = returns.cov()
    elif method == "ledoit_wolf":
        lw = LedoitWolf().fit(returns.values)
        cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    return cov * TRADING_DAYS


def risk_free_rate(conn: sqlite3.Connection, start: str, end: str | None = None) -> float:
    """Average 3-month T-bill rate over the period, as a decimal (4.5% -> 0.045)."""
    query = "SELECT AVG(rate) FROM risk_free_rates WHERE date >= ?"
    params: list = [start]
    if end:
        query += " AND date <= ?"
        params.append(end)
    value = conn.execute(query, params).fetchone()[0]
    return (value or 0.0) / 100


# ---------------------------------------------------------------------------
# Quick look at the data – run this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    START = "2023-01-01"

    with sqlite3.connect(DB_PATH) as conn:
        tickers = get_stock_tickers(conn)
        prices = load_prices(conn, tickers, start=START)
        rets = daily_returns(prices)
        mu = expected_returns(rets)
        sigma = covariance_matrix(rets, method="ledoit_wolf")
        rf = risk_free_rate(conn, START)

    vol = pd.Series(np.sqrt(np.diag(sigma)), index=sigma.index)
    summary = pd.DataFrame({"exp_return": mu, "volatility": vol})
    summary["sharpe"] = (summary["exp_return"] - rf) / summary["volatility"]

    print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()} "
          f"({len(rets)} trading days, {len(tickers)} stocks)")
    print(f"Risk-free rate (avg): {rf:.2%}\n")
    print(summary.sort_values("sharpe", ascending=False).round(3).to_string())

    corr = rets.corr()
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    print("\nMost correlated pairs:")
    print(pairs.sort_values(ascending=False).head(5).round(2).to_string())
    print("\nLeast correlated pairs:")
    print(pairs.sort_values().head(5).round(2).to_string())
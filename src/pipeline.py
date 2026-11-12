"""
Daily data pipeline: pulls adjusted prices (yfinance) and the
risk-free rate (FRED) into portfolio.db. Safe to re-run: only new
rows are inserted.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "portfolio.db"
DEFAULT_START = "2015-01-01"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"


def get_tickers(conn: sqlite3.Connection) -> list[str]:
    """Return all tickers in the assets table (stocks + benchmark)."""
    rows = conn.execute("SELECT ticker FROM assets").fetchall()
    return [row[0] for row in rows]


def get_last_date(conn: sqlite3.Connection, table: str) -> str | None:
    """Return the latest date stored in `table`, or None if it is empty."""
    row = conn.execute(f"SELECT max(date) FROM {table}").fetchone()

    return row[0]


def fetch_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """
    Download daily prices for all tickers from `start` until today.
    Returns a long-format DataFrame with columns:
        ticker, date ('YYYY-MM-DD'), adj_close, volume
    """
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    long = data[["Close","Volume"]].stack(level=1).reset_index()
    long = long.rename(columns={
    "Date": "date",
    "Ticker": "ticker",
    "Close": "adj_close",
    "Volume": "volume",
})
    long["date"] = long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    long = long.dropna(subset=["adj_close"])
    return long[["ticker", "date", "adj_close", "volume"]]


def save_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert rows into prices, skipping duplicates. Return rows inserted."""
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO prices (ticker, date, adj_close, volume) VALUES (?, ?, ?, ?)",
        df.itertuples(index=False, name=None),
    )
    return conn.total_changes - before


def fetch_risk_free(start: str) -> pd.DataFrame:
    """Download the 3-month T-bill rate (annual %) from FRED. Columns: date, rate."""
    df = pd.read_csv(FRED_URL)
    df.columns = ["date", "rate"]
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna()
    return df[df["date"] >= start]


def save_risk_free(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert rows into risk_free_rates, skipping duplicates. Return rows inserted."""
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO risk_free_rates (date, rate) VALUES (?, ?)",
        df.itertuples(index=False, name=None),
    )
    return conn.total_changes - before


# ---------------------------------------------------------------------------
# Orchestration (complete – no need to change)
# ---------------------------------------------------------------------------

def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("INSERT INTO pipeline_runs (status) VALUES ('running')")
        run_id = cur.lastrowid
        conn.commit()

        try:
            tickers = get_tickers(conn)
            start = get_last_date(conn, "prices") or DEFAULT_START
            prices = fetch_prices(tickers, start)
            n_prices = save_prices(conn, prices)

            rf_start = get_last_date(conn, "risk_free_rates") or DEFAULT_START
            rates = fetch_risk_free(rf_start)
            n_rates = save_risk_free(conn, rates)

            total = n_prices + n_rates
            conn.execute(
                "UPDATE pipeline_runs SET status='success', finished_at=?, rows_inserted=? "
                "WHERE run_id=?",
                (datetime.now().isoformat(timespec="seconds"), total, run_id),
            )
            conn.commit()
            print(f"Done: {n_prices} price rows, {n_rates} rate rows.")

        except Exception as e:
            conn.rollback()
            conn.execute(
                "UPDATE pipeline_runs SET status='failed', finished_at=?, error_message=? "
                "WHERE run_id=?",
                (datetime.now().isoformat(timespec="seconds"), str(e), run_id),
            )
            conn.commit()
            raise


if __name__ == "__main__":
    main()
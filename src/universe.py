"""
Stock universe: the list of every US-listed stock the user can pick from,
and adding a new stock (with its full price history) to the database.

The list comes from the official NASDAQ Trader symbol files, which cover
NASDAQ, NYSE and the other US exchanges. It is cached in the database and
refreshed once a week.
"""
import re
import sqlite3
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

import pipeline

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
REFRESH_DAYS = 7

# Securities that are not ordinary shares
EXCLUDE = re.compile(r"warrant|\bunits?\b|\bright(s)?\b|preferred|\bnotes?\b|debenture|%", re.IGNORECASE)
# Trailing descriptions to drop from the company name
SUFFIX = re.compile(
    r"\s*[-,]?\s*(class [a-z] )?(common stock|common shares|ordinary shares|capital stock"
    r"|american depositary shares.*|shares of beneficial interest).*$",
    re.IGNORECASE,
)

DDL = """
CREATE TABLE IF NOT EXISTS ticker_universe (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    exchange    TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (date('now'))
);
"""

EXCHANGES = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)


def clean_name(raw: str) -> str:
    name = SUFFIX.sub("", raw).strip(" -,")
    return name or raw


def download_universe() -> pd.DataFrame:
    """All US common stocks and ADRs. Columns: ticker, name, exchange."""
    nasdaq = pd.read_csv(NASDAQ_URL, sep="|", dtype=str)
    nasdaq = nasdaq[(nasdaq["Test Issue"] == "N") & (nasdaq["ETF"] == "N")]
    nasdaq = pd.DataFrame({"ticker": nasdaq["Symbol"], "name": nasdaq["Security Name"], "exchange": "NASDAQ"})

    other = pd.read_csv(OTHER_URL, sep="|", dtype=str)
    other = other[(other["Test Issue"] == "N") & (other["ETF"] == "N")]
    other = pd.DataFrame({
        "ticker": other["ACT Symbol"],
        "name": other["Security Name"],
        "exchange": other["Exchange"].map(EXCHANGES).fillna("Other"),
    })

    df = pd.concat([nasdaq, other]).dropna()
    df = df[~df["ticker"].str.contains(r"[\$\^ ]|File Creation", regex=True)]
    df = df[~df["name"].str.contains(EXCLUDE)]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)   # BRK.B -> BRK-B (yfinance format)
    df["name"] = df["name"].map(clean_name)
    return df.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def refresh_universe(conn: sqlite3.Connection) -> bool:
    """Re-download the stock list if it is missing or older than a week. Returns False if offline."""
    ensure_tables(conn)
    last = conn.execute("SELECT MAX(updated_at) FROM ticker_universe").fetchone()[0]
    if last and date.fromisoformat(last) > date.today() - timedelta(days=REFRESH_DAYS):
        return True
    try:
        df = download_universe()
    except Exception:
        return False
    if len(df) < 1000:          # something went wrong with the download; keep the old list
        return False
    conn.execute("DELETE FROM ticker_universe")
    conn.executemany("INSERT INTO ticker_universe (ticker, name, exchange) VALUES (?, ?, ?)",
                     df.itertuples(index=False, name=None))
    conn.commit()
    return True


def load_universe(conn: sqlite3.Connection) -> pd.DataFrame:
    """Stocks the user can choose: the downloaded list plus everything already in the database."""
    ensure_tables(conn)
    df = pd.read_sql_query(
        """SELECT ticker, name FROM assets WHERE asset_type = 'stock'
           UNION
           SELECT ticker, name FROM ticker_universe
           WHERE ticker NOT IN (SELECT ticker FROM assets)""",
        conn,
    )
    return df.drop_duplicates("ticker").set_index("ticker").sort_index()


# ---------------------------------------------------------------------------
# Adding a new stock
# ---------------------------------------------------------------------------

def _download_history(ticker: str) -> pd.DataFrame:
    """Full daily history of one ticker. Columns: ticker, date, adj_close, volume."""
    kwargs = dict(start=pipeline.DEFAULT_START, auto_adjust=True, progress=False)
    try:
        data = yf.download(ticker, multi_level_index=False, **kwargs)
    except TypeError:                       # older yfinance without multi_level_index
        data = yf.download(ticker, **kwargs)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty or "Close" not in data:
        return pd.DataFrame()
    df = data[["Close", "Volume"]].dropna(subset=["Close"]).reset_index()
    df.columns = ["date", "adj_close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df.insert(0, "ticker", ticker)
    return df[df["adj_close"] > 0]


def _sector(ticker: str) -> str:
    try:
        return yf.Ticker(ticker).info.get("sector") or "Other"
    except Exception:
        return "Other"


def add_stock(conn: sqlite3.Connection, ticker: str, name: str) -> int:
    """
    Download a stock's history and save it (assets + prices) in one transaction.
    Returns the number of price rows saved. Raises ValueError if there is no data.
    """
    history = _download_history(ticker)
    if len(history) < 60:
        raise ValueError(f"Not enough price history for {ticker}.")
    with conn:
        conn.execute("INSERT OR IGNORE INTO assets (ticker, name, sector, asset_type) VALUES (?, ?, ?, 'stock')",
                     (ticker, name, _sector(ticker)))
        return pipeline.save_prices(conn, history)

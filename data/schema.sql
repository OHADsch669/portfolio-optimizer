-- Portfolio Optimizer – database schema (SQLite)
-- Run: sqlite3 portfolio.db < schema.sql

PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. Reference data
-- ============================================================

CREATE TABLE IF NOT EXISTS assets (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    sector      TEXT NOT NULL,
    asset_type  TEXT NOT NULL DEFAULT 'stock'
                CHECK (asset_type IN ('stock', 'benchmark')),
    added_at    TEXT NOT NULL DEFAULT (date('now'))
);

-- ============================================================
-- 2. Market data (filled by the daily pipeline)
-- ============================================================

-- One row per ticker per trading day. Store prices, not returns:
-- returns are derived in Python so there is a single source of truth.
CREATE TABLE IF NOT EXISTS prices (
    ticker      TEXT NOT NULL REFERENCES assets(ticker),
    date        TEXT NOT NULL,              -- ISO 'YYYY-MM-DD'
    adj_close   REAL NOT NULL CHECK (adj_close > 0),
    volume      INTEGER,
    PRIMARY KEY (ticker, date)              -- prevents duplicates on re-runs
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

-- 3-month US Treasury bill rate (FRED: DTB3), annual %, for Sharpe ratio
CREATE TABLE IF NOT EXISTS risk_free_rates (
    date        TEXT PRIMARY KEY,
    rate        REAL NOT NULL
);

-- Log of every pipeline execution (monitoring / data freshness)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at   TEXT,
    status        TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    rows_inserted INTEGER DEFAULT 0,
    error_message TEXT
);

-- ============================================================
-- 3. Optimization results
-- ============================================================

CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    strategy        TEXT NOT NULL
                    CHECK (strategy IN ('max_sharpe', 'min_variance', 'risk_parity', 'equal_weight')),
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    max_weight      REAL CHECK (max_weight > 0 AND max_weight <= 1),
    cov_method      TEXT NOT NULL DEFAULT 'sample'
                    CHECK (cov_method IN ('sample', 'ledoit_wolf')),
    exp_return      REAL,                   -- annualized
    volatility      REAL,                   -- annualized
    sharpe          REAL
);

CREATE TABLE IF NOT EXISTS run_weights (
    run_id      INTEGER NOT NULL REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    ticker      TEXT    NOT NULL REFERENCES assets(ticker),
    weight      REAL    NOT NULL CHECK (weight >= 0 AND weight <= 1),
    PRIMARY KEY (run_id, ticker)
);

-- ============================================================
-- 4. Backtests
-- ============================================================

CREATE TABLE IF NOT EXISTS backtests (
    backtest_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    strategy            TEXT NOT NULL,
    start_date          TEXT NOT NULL,
    end_date            TEXT NOT NULL,
    lookback_days       INTEGER NOT NULL,    -- estimation window
    rebalance_freq      TEXT NOT NULL CHECK (rebalance_freq IN ('monthly', 'quarterly')),
    transaction_cost_bps REAL NOT NULL DEFAULT 10,
    max_weight          REAL,
    cov_method          TEXT NOT NULL DEFAULT 'sample',
    -- result metrics
    cagr                REAL,
    volatility          REAL,
    sharpe              REAL,
    max_drawdown        REAL
);

CREATE TABLE IF NOT EXISTS backtest_values (
    backtest_id     INTEGER NOT NULL REFERENCES backtests(backtest_id) ON DELETE CASCADE,
    date            TEXT    NOT NULL,
    portfolio_value REAL    NOT NULL,       -- starts at 1.0
    PRIMARY KEY (backtest_id, date)
);

-- ============================================================
-- 5. Seed: 30 stocks (10 sectors x 3) + benchmark
-- ============================================================

INSERT OR IGNORE INTO assets (ticker, name, sector, asset_type) VALUES
    ('AAPL',  'Apple',                  'Technology',             'stock'),
    ('MSFT',  'Microsoft',              'Technology',             'stock'),
    ('NVDA',  'NVIDIA',                 'Technology',             'stock'),
    ('GOOGL', 'Alphabet',               'Communication Services', 'stock'),
    ('META',  'Meta Platforms',         'Communication Services', 'stock'),
    ('VZ',    'Verizon',                'Communication Services', 'stock'),
    ('AMZN',  'Amazon',                 'Consumer Discretionary', 'stock'),
    ('HD',    'Home Depot',             'Consumer Discretionary', 'stock'),
    ('MCD',   'McDonald''s',            'Consumer Discretionary', 'stock'),
    ('PG',    'Procter & Gamble',       'Consumer Staples',       'stock'),
    ('KO',    'Coca-Cola',              'Consumer Staples',       'stock'),
    ('WMT',   'Walmart',                'Consumer Staples',       'stock'),
    ('JNJ',   'Johnson & Johnson',      'Healthcare',             'stock'),
    ('UNH',   'UnitedHealth',           'Healthcare',             'stock'),
    ('LLY',   'Eli Lilly',              'Healthcare',             'stock'),
    ('JPM',   'JPMorgan Chase',         'Financials',             'stock'),
    ('BAC',   'Bank of America',        'Financials',             'stock'),
    ('BRK-B', 'Berkshire Hathaway',     'Financials',             'stock'),
    ('CAT',   'Caterpillar',            'Industrials',            'stock'),
    ('HON',   'Honeywell',              'Industrials',            'stock'),
    ('UNP',   'Union Pacific',          'Industrials',            'stock'),
    ('XOM',   'Exxon Mobil',            'Energy',                 'stock'),
    ('CVX',   'Chevron',                'Energy',                 'stock'),
    ('COP',   'ConocoPhillips',         'Energy',                 'stock'),
    ('NEE',   'NextEra Energy',         'Utilities',              'stock'),
    ('DUK',   'Duke Energy',            'Utilities',              'stock'),
    ('SO',    'Southern Company',       'Utilities',              'stock'),
    ('LIN',   'Linde',                  'Materials',              'stock'),
    ('SHW',   'Sherwin-Williams',       'Materials',              'stock'),
    ('NEM',   'Newmont',                'Materials',              'stock'),
    ('SPY',   'SPDR S&P 500 ETF',       'Benchmark',              'benchmark');

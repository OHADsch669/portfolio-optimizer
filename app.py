"""
Portfolio Optimizer – Streamlit dashboard.
Run from the project root:  streamlit run app.py
"""
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import optimizer as opt
import stats

ACCENT, ORANGE, GRAY = "#1D4ED8", "#C2410C", "#9CA3AF"
STRATEGIES = {
    "Max Sharpe ratio": "max_sharpe",
    "Minimum variance": "min_variance",
    "Risk parity": "risk_parity",
    "Equal weight (1/N)": "equal_weight",
}
BACKTEST_NAMES = {
    "equal_weight": "Equal weight", "min_variance": "Min variance", "max_sharpe": "Max Sharpe",
    "risk_parity": "Risk parity", "benchmark_spy": "SPY (benchmark)",
}

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Data access (cached so the app stays fast)
# ---------------------------------------------------------------------------

def connect() -> sqlite3.Connection:
    return sqlite3.connect(stats.DB_PATH)


@st.cache_data
def load_assets() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT ticker, name, sector, asset_type FROM assets", conn).set_index("ticker")


@st.cache_data
def last_price_date() -> date:
    with connect() as conn:
        return date.fromisoformat(conn.execute("SELECT MAX(date) FROM prices").fetchone()[0])


@st.cache_data
def load_returns(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    with connect() as conn:
        return stats.daily_returns(stats.load_prices(conn, list(tickers), start, end))


@st.cache_data
def load_rf(start: str, end: str) -> float:
    with connect() as conn:
        return stats.risk_free_rate(conn, start, end)


@st.cache_data
def compute_weights(strategy: str, mu: np.ndarray, sigma: np.ndarray, rf: float, cap: float) -> np.ndarray:
    if strategy == "equal_weight":
        return opt.equal_weight(len(mu))
    if strategy == "min_variance":
        return opt.min_variance(sigma, cap)
    if strategy == "max_sharpe":
        return opt.max_sharpe(mu, sigma, rf, cap)
    return opt.risk_parity(sigma)


@st.cache_data
def compute_frontier(mu: np.ndarray, sigma: np.ndarray, cap: float) -> pd.DataFrame:
    return opt.efficient_frontier(mu, sigma, cap, points=25)


@st.cache_data
def load_backtests() -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_ids = "SELECT MAX(backtest_id) FROM backtests GROUP BY strategy"
    with connect() as conn:
        summary = pd.read_sql_query(f"SELECT * FROM backtests WHERE backtest_id IN ({latest_ids})", conn)
        values = pd.read_sql_query(
            f"""SELECT b.strategy, v.date, v.portfolio_value
                FROM backtest_values v JOIN backtests b USING (backtest_id)
                WHERE v.backtest_id IN ({latest_ids})""",
            conn, parse_dates=["date"],
        )
    return summary, values


def pct_axis(fig: go.Figure, x: bool = True, y: bool = True) -> go.Figure:
    if x:
        fig.update_xaxes(tickformat=".0%")
    if y:
        fig.update_yaxes(tickformat=".0%")
    return fig


# ---------------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------------

assets = load_assets()
stocks = assets[assets["asset_type"] == "stock"]
max_date = last_price_date()

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect(
        "Stocks", options=list(stocks.index), default=list(stocks.index),
        format_func=lambda t: f"{t} – {stocks.loc[t, 'name']}",
    )
    c1, c2 = st.columns(2)
    start = c1.date_input("From", date(2023, 1, 1), min_value=date(2015, 1, 2), max_value=max_date)
    end = c2.date_input("To", max_date, min_value=date(2015, 1, 2), max_value=max_date)
    strategy_label = st.radio("Strategy", list(STRATEGIES))
    cap = st.slider("Max weight per stock", 5, 100, 20, step=5, format="%d%%") / 100
    use_lw = st.checkbox("Use Ledoit-Wolf covariance", value=True,
                         help="Shrinks the covariance matrix toward a stable target to reduce estimation noise.")
    st.caption(f"Prices updated through {max_date:%b %d, %Y}")

st.title("Portfolio Optimizer")

if len(tickers) < 2:
    st.warning("Select at least two stocks.")
    st.stop()
if cap * len(tickers) < 1:
    st.error(f"With {len(tickers)} stocks, the max weight must be at least {1 / len(tickers):.0%}.")
    st.stop()
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

returns = load_returns(tuple(tickers), str(start), str(end))
if len(returns) < 60:
    st.warning("Pick a period of at least three months for meaningful estimates.")
    st.stop()

strategy = STRATEGIES[strategy_label]
mu = stats.expected_returns(returns).values
sigma = stats.covariance_matrix(returns, method="ledoit_wolf" if use_lw else "sample").values
rf = load_rf(str(start), str(end))

try:
    w = compute_weights(strategy, mu, sigma, rf, cap)
except RuntimeError as e:
    st.error(f"The optimizer could not find a solution: {e}")
    st.stop()

port_daily = pd.Series(returns.values @ w, index=returns.index)
equity = (1 + port_daily).cumprod()
max_dd = (equity / equity.cummax() - 1).min()
stock_vol = np.sqrt(np.diag(sigma))
rc = opt.risk_contributions(w, sigma)

tab_build, tab_bt, tab_risk, tab_method = st.tabs(
    ["Portfolio Builder", "Backtest", "Risk Analysis", "Methodology"])


# ---------------------------------------------------------------------------
# Tab 1 – Portfolio Builder
# ---------------------------------------------------------------------------

with tab_build:
    st.subheader(f"{strategy_label} portfolio")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected return (annual)", f"{opt.portfolio_return(w, mu):.1%}")
    k2.metric("Volatility (annual)", f"{opt.portfolio_volatility(w, sigma):.1%}")
    k3.metric("Sharpe ratio", f"{opt.sharpe_ratio(w, mu, sigma, rf):.2f}")
    k4.metric("Max drawdown (in period)", f"{max_dd:.1%}")
    st.caption("In-sample figures: estimated and measured on the same period. "
               "See the Backtest tab for out-of-sample results.")
    if strategy == "risk_parity" and w.max() > cap + 1e-6:
        st.info("Risk parity sets weights by risk contribution, so the max-weight cap is not applied.")

    left, right = st.columns([3, 2])

    with left:
        fig = go.Figure()
        try:
            frontier = compute_frontier(mu, sigma, cap)
            fig.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["return"], mode="lines",
                                     name="Efficient frontier", line=dict(color="#111827", width=2)))
        except RuntimeError:
            st.caption("The efficient frontier could not be computed for these settings.")
        fig.add_trace(go.Scatter(x=stock_vol, y=mu, mode="markers", name="Single stocks", text=tickers,
                                 marker=dict(color=GRAY, size=8),
                                 hovertemplate="%{text}<br>vol %{x:.1%}<br>return %{y:.1%}<extra></extra>"))
        w_mv = opt.min_variance(sigma, cap)
        fig.add_trace(go.Scatter(x=[opt.portfolio_volatility(w_mv, sigma)], y=[opt.portfolio_return(w_mv, mu)],
                                 mode="markers", name="Min variance",
                                 marker=dict(color=ORANGE, size=12, symbol="square")))
        fig.add_trace(go.Scatter(x=[opt.portfolio_volatility(w, sigma)], y=[opt.portfolio_return(w, mu)],
                                 mode="markers", name="Your portfolio",
                                 marker=dict(color=ACCENT, size=16, line=dict(color="white", width=2))))
        fig.update_layout(title="Efficient frontier", xaxis_title="Volatility (annualized)",
                          yaxis_title="Expected return", height=440, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(pct_axis(fig), width="stretch")

    with right:
        alloc = pd.Series(w, index=tickers)
        alloc = alloc[alloc > 0.001].sort_values()
        fig = px.bar(x=alloc.values, y=alloc.index, orientation="h", title="Allocation",
                     labels={"x": "Weight", "y": ""}, color_discrete_sequence=[ACCENT])
        fig.update_layout(height=440)
        st.plotly_chart(pct_axis(fig, y=False), width="stretch")

    holdings = pd.DataFrame({
        "Ticker": tickers,
        "Company": stocks.loc[tickers, "name"].values,
        "Sector": stocks.loc[tickers, "sector"].values,
        "Weight %": w * 100,
        "Exp. return %": mu * 100,
        "Volatility %": stock_vol * 100,
        "Risk contribution %": rc * 100,
    })
    holdings = holdings[holdings["Weight %"] > 0.1].sort_values("Weight %", ascending=False)
    st.markdown("#### Holdings detail")
    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(holdings, hide_index=True, width="stretch",
                 column_config={c: pct for c in holdings.columns if c.endswith("%")})
    st.download_button("Download CSV", holdings.to_csv(index=False).encode(),
                       file_name=f"portfolio_{strategy}.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Tab 2 – Backtest
# ---------------------------------------------------------------------------

with tab_bt:
    summary, values = load_backtests()
    if summary.empty:
        st.info("No backtest results yet. Run `python src/backtest.py` once, then refresh.")
    else:
        st.subheader("Walk-forward backtest")
        row = summary.iloc[0]
        st.caption(f"{row.start_date} to {row.end_date} · all 30 stocks · monthly rebalance · "
                   f"{row.lookback_days}-day lookback · {row.transaction_cost_bps:.0f} bps costs · "
                   f"cap {row.max_weight:.0%}. Weights each month use only data available at that time.")

        values["Strategy"] = values["strategy"].map(BACKTEST_NAMES)
        fig = px.line(values, x="date", y="portfolio_value", color="Strategy",
                      title="Growth of $1", labels={"date": "", "portfolio_value": "Value ($)"})
        fig.update_layout(height=440, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

        table = summary.assign(Strategy=summary["strategy"].map(BACKTEST_NAMES))
        table = table.set_index("Strategy")[["cagr", "volatility", "sharpe", "max_drawdown"]]
        table = table.sort_values("sharpe", ascending=False)
        table.columns = ["CAGR", "Volatility", "Sharpe", "Max drawdown"]
        st.dataframe(table.style.format({"CAGR": "{:.1%}", "Volatility": "{:.1%}",
                                         "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}"}),
                     width="stretch")

        dd = values.sort_values("date").copy()
        dd["Drawdown"] = dd.groupby("Strategy")["portfolio_value"].transform(lambda v: v / v.cummax() - 1)
        fig = px.line(dd, x="date", y="Drawdown", color="Strategy", title="Drawdown from previous peak",
                      labels={"date": ""})
        fig.update_layout(height=340, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(pct_axis(fig, x=False), width="stretch")


# ---------------------------------------------------------------------------
# Tab 3 – Risk Analysis
# ---------------------------------------------------------------------------

with tab_risk:
    left, right = st.columns(2)
    with left:
        sectors = (pd.Series(w, index=stocks.loc[tickers, "sector"].values)
                   .groupby(level=0).sum().sort_values())
        fig = px.bar(x=sectors.values, y=sectors.index, orientation="h", title="Sector exposure",
                     labels={"x": "Weight", "y": ""}, color_discrete_sequence=[ACCENT])
        st.plotly_chart(pct_axis(fig, y=False), width="stretch")
    with right:
        comp = pd.DataFrame({"Ticker": tickers, "Weight": w, "Risk contribution": rc})
        comp = comp[comp["Weight"] > 0.001].sort_values("Weight", ascending=False)
        comp = comp.melt(id_vars="Ticker", var_name="Measure", value_name="Share")
        fig = px.bar(comp, x="Ticker", y="Share", color="Measure", barmode="group",
                     title="Share of money vs share of risk",
                     color_discrete_map={"Weight": ACCENT, "Risk contribution": ORANGE})
        st.plotly_chart(pct_axis(fig, x=False), width="stretch")

    corr = returns.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation of daily returns", aspect="auto")
    fig.update_layout(height=640)
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Tab 4 – Methodology
# ---------------------------------------------------------------------------

with tab_method:
    st.markdown("""
### What this tool does
Given a set of stocks, it decides **how to split money between them** for the best return per unit of risk,
and tests which allocation method actually works out of sample.

### Pipeline
1. **Data** – daily adjusted prices (yfinance) and the 3-month T-bill rate (FRED), stored in SQLite and updated incrementally.
2. **Estimates** – expected returns (annualized historical mean) and the covariance matrix (sample or Ledoit-Wolf shrinkage).
3. **Optimization** – weights sum to 1, no short selling, optional cap per stock; solved with SLSQP (SciPy).
4. **Backtest** – walk-forward: every month, weights are re-estimated from the previous two years only, with transaction costs.

### Strategies
- **Max Sharpe** – highest excess return per unit of risk. Most sensitive to errors in expected returns.
- **Minimum variance** – lowest possible risk; ignores expected returns.
- **Risk parity** – every stock contributes the same share of total risk.
- **Equal weight (1/N)** – the naive benchmark that is notoriously hard to beat.

### Limitations
- **Survivorship bias:** the stock universe was chosen in 2026 from today's large companies, which flatters past results.
- **Expected returns are noisy:** historical averages are weak forecasts; see how Max Sharpe performs out of sample.
- Results are for analysis and education only and are **not investment advice**.
""")

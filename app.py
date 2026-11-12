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
import pipeline
import stats

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Theme-aware colors (light / dark mode)
# ---------------------------------------------------------------------------

try:
    IS_DARK = st.context.theme.type == "dark"
except AttributeError:
    IS_DARK = False

ACCENT = "#60A5FA" if IS_DARK else "#1D4ED8"
ORANGE = "#FB923C" if IS_DARK else "#C2410C"
GRAY = "#6B7280" if IS_DARK else "#9CA3AF"
INK = "#E5E7EB" if IS_DARK else "#111827"

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

# Plain-language explanations, shown when hovering over a column or metric
HELP = {
    "ticker": "The stock's short code on the exchange.",
    "company": "The company behind the stock.",
    "sector": "The part of the economy the company belongs to.",
    "weight": "The share of your money to put in this stock. All weights add up to 100%.",
    "exp_return": "How much this stock gained per year, on average, in the period you picked. "
                  "The optimizer uses it as its best guess for the future.",
    "volatility": "How much the price jumps up and down in a typical year. Higher means riskier.",
    "risk_contrib": "How much of the whole portfolio's ups and downs come from this stock. "
                    "All rows add up to 100%.",
    "port_return": "The average yearly gain this portfolio would have had in the period you picked.",
    "port_vol": "How much the portfolio's value swings in a typical year. Lower means a smoother ride.",
    "sharpe": "Return earned for each unit of risk, above what a risk-free government bill pays. "
              "Above 1 is very good.",
    "max_dd": "The biggest drop from a peak to a low: the worst loss you would have felt along the way.",
    "cagr": "The average yearly growth that actually happened, rebalancing every month "
            "using only information available at the time.",
    "bt_vol": "How much the strategy's value swung per year in the test.",
    "last_price": "The most recent closing price, adjusted for dividends and splits.",
    "period_return": "Total gain or loss over the period shown in the chart.",
}

# ---------------------------------------------------------------------------
# Look and feel: fade-in on every tab switch, card-style metrics
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: none; }
}
[role="tabpanel"], [data-baseweb="tab-panel"] { animation: fadeIn 0.45s ease-out; }
[data-testid="stMetric"] {
  border: 1px solid rgba(128, 128, 128, 0.25);
  border-radius: 12px;
  padding: 14px 18px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12);
}
@media (prefers-reduced-motion: reduce) {
  [role="tabpanel"], [data-baseweb="tab-panel"] { animation: none; }
  [data-testid="stMetric"], [data-testid="stMetric"]:hover { transition: none; transform: none; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data access (cached so the app stays fast)
# ---------------------------------------------------------------------------

def connect() -> sqlite3.Connection:
    return sqlite3.connect(stats.DB_PATH)


@st.cache_data(ttl=6 * 3600, show_spinner="Checking for new market data...")
def refresh_market_data() -> bool:
    """Run the incremental pipeline at most every 6 hours. Falls back to stored data on failure."""
    try:
        pipeline.main()
        return True
    except Exception:
        return False


@st.cache_data
def load_assets() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT ticker, name, sector, asset_type FROM assets", conn).set_index("ticker")


@st.cache_data(ttl=3600)
def last_price_date() -> date:
    with connect() as conn:
        return date.fromisoformat(conn.execute("SELECT MAX(date) FROM prices").fetchone()[0])


@st.cache_data
def load_returns(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    with connect() as conn:
        return stats.daily_returns(stats.load_prices(conn, list(tickers), start, end))


@st.cache_data
def load_price_history(ticker: str) -> pd.DataFrame:
    """Full price history of one ticker plus SPY, for the stock explorer."""
    tickers = [ticker] if ticker == "SPY" else [ticker, "SPY"]
    with connect() as conn:
        return stats.load_prices(conn, tickers)


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


def pct_col(label: str, help_key: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="%.1f%%", help=HELP[help_key])


def text_col(label: str, help_key: str) -> st.column_config.TextColumn:
    return st.column_config.TextColumn(label, help=HELP[help_key])


# ---------------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------------

data_is_fresh = refresh_market_data()
assets = load_assets()
stocks = assets[assets["asset_type"] == "stock"]
max_date = last_price_date()

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect(
        "Stocks", options=list(stocks.index), default=list(stocks.index),
        format_func=lambda t: f"{t} – {stocks.loc[t, 'name']}",
        help="The stocks the optimizer may split your money between.",
    )
    c1, c2 = st.columns(2)
    start = c1.date_input("From", date(2023, 1, 1), min_value=date(2015, 1, 2), max_value=max_date,
                          help="Start of the history used to estimate returns and risk.")
    end = c2.date_input("To", max_date, min_value=date(2015, 1, 2), max_value=max_date)
    strategy_label = st.radio(
        "Strategy", list(STRATEGIES),
        help="Max Sharpe: best return for the risk. Minimum variance: lowest risk. "
             "Risk parity: every stock adds the same risk. Equal weight: the same amount in each stock.",
    )
    cap = st.slider("Max weight per stock", 5, 100, 20, step=5, format="%d%%",
                    help="No single stock may get more than this share of the money.") / 100
    use_lw = st.checkbox("Use Ledoit-Wolf covariance", value=True,
                         help="A more stable way to estimate how stocks move together. "
                              "Reduces the effect of random noise in the data.")
    st.caption(f"Prices updated through {max_date:%b %d, %Y}"
               + (" · refreshed automatically" if data_is_fresh else " · using stored data"))
    st.caption("🌗 Light / dark mode: ⋮ menu (top right) → Settings → Theme")

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
weights_by_ticker = pd.Series(w, index=tickers)

tab_build, tab_explore, tab_bt, tab_risk, tab_method = st.tabs(
    ["Portfolio Builder", "Stock Explorer", "Backtest", "Risk Analysis", "Methodology"])


# ---------------------------------------------------------------------------
# Tab 1 – Portfolio Builder
# ---------------------------------------------------------------------------

with tab_build:
    st.subheader(f"{strategy_label} portfolio")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected return (annual)", f"{opt.portfolio_return(w, mu):.1%}", help=HELP["port_return"])
    k2.metric("Volatility (annual)", f"{opt.portfolio_volatility(w, sigma):.1%}", help=HELP["port_vol"])
    k3.metric("Sharpe ratio", f"{opt.sharpe_ratio(w, mu, sigma, rf):.2f}", help=HELP["sharpe"])
    k4.metric("Max drawdown (in period)", f"{max_dd:.1%}", help=HELP["max_dd"])
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
                                     name="Efficient frontier", line=dict(color=INK, width=2)))
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
        st.caption("The line shows the best return possible for each level of risk. "
                   "Grey dots are single stocks: almost all sit below the line, which is why we diversify.")

    with right:
        alloc = weights_by_ticker[weights_by_ticker > 0.001].sort_values()
        fig = px.bar(x=alloc.values, y=alloc.index, orientation="h", title="Allocation",
                     labels={"x": "Weight", "y": ""}, color_discrete_sequence=[ACCENT])
        fig.update_traces(hovertemplate="%{y}: %{x:.1%} of your money<extra></extra>")
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
    st.caption("Hover over a column title to see what it means.")
    st.dataframe(holdings, hide_index=True, width="stretch", column_config={
        "Ticker": text_col("Ticker", "ticker"),
        "Company": text_col("Company", "company"),
        "Sector": text_col("Sector", "sector"),
        "Weight %": pct_col("Weight %", "weight"),
        "Exp. return %": pct_col("Exp. return %", "exp_return"),
        "Volatility %": pct_col("Volatility %", "volatility"),
        "Risk contribution %": pct_col("Risk contribution %", "risk_contrib"),
    })
    st.download_button("Download CSV", holdings.to_csv(index=False).encode(),
                       file_name=f"portfolio_{strategy}.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Tab 2 – Stock Explorer
# ---------------------------------------------------------------------------

with tab_explore:
    options = list(stocks.index) + ["SPY"]
    choice = st.selectbox(
        "Search a stock", options, index=None,
        placeholder="Type a ticker or company name, e.g. Apple or NVDA",
        format_func=lambda t: f"{t} – {assets.loc[t, 'name']}",
    )
    if choice is None:
        st.info("Pick a stock above to see its price history and key numbers.")
    else:
        history = load_price_history(choice)
        info = assets.loc[choice]
        st.subheader(f"{info['name']} ({choice})")
        in_portfolio = weights_by_ticker.get(choice, 0.0)
        st.caption(f"{info['sector']} · "
                   + (f"{in_portfolio:.1%} of your current portfolio" if in_portfolio > 0.001
                      else "not in your current portfolio"))

        periods = {"1M": 21, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "All": None}
        period = st.segmented_control("Period", list(periods), default="1Y")
        days = periods[period or "1Y"]
        view = history.iloc[-(days + 1):] if days else history
        price = view[choice].dropna()

        daily = price.pct_change().dropna()
        drawdown = (price / price.cummax() - 1).min()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last price", f"${price.iloc[-1]:,.2f}", help=HELP["last_price"])
        m2.metric(f"Return ({period or '1Y'})", f"{price.iloc[-1] / price.iloc[0] - 1:+.1%}",
                  help=HELP["period_return"])
        m3.metric("Volatility (annual)", f"{daily.std() * np.sqrt(stats.TRADING_DAYS):.1%}",
                  help=HELP["volatility"])
        m4.metric("Max drawdown", f"{drawdown:.1%}", help=HELP["max_dd"])

        compare = choice != "SPY" and st.toggle("Compare with the S&P 500 (SPY)",
                                                help="Both lines start at 100, so you can see "
                                                     "which grew more over the same period.")
        fig = go.Figure()
        if compare:
            for t, color in [(choice, ACCENT), ("SPY", GRAY)]:
                s = view[t].dropna()
                fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, mode="lines", name=t,
                                         line=dict(color=color, width=2)))
            fig.update_layout(yaxis_title="Growth of 100")
        else:
            fig.add_trace(go.Scatter(x=price.index, y=price, mode="lines", name=choice, fill="tozeroy",
                                     line=dict(color=ACCENT, width=2),
                                     fillcolor="rgba(96,165,250,0.15)" if IS_DARK else "rgba(29,78,216,0.08)",
                                     hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>"))
            fig.update_yaxes(range=[price.min() * 0.95, price.max() * 1.03], tickprefix="$")
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.15), xaxis_title="")
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Tab 3 – Backtest
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

        table = pd.DataFrame({
            "Strategy": summary["strategy"].map(BACKTEST_NAMES),
            "CAGR %": summary["cagr"] * 100,
            "Volatility %": summary["volatility"] * 100,
            "Sharpe": summary["sharpe"],
            "Max drawdown %": summary["max_drawdown"] * 100,
        }).sort_values("Sharpe", ascending=False)
        st.caption("Hover over a column title to see what it means.")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "CAGR %": pct_col("CAGR %", "cagr"),
            "Volatility %": pct_col("Volatility %", "bt_vol"),
            "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f", help=HELP["sharpe"]),
            "Max drawdown %": pct_col("Max drawdown %", "max_dd"),
        })

        dd = values.sort_values("date").copy()
        dd["Drawdown"] = dd.groupby("Strategy")["portfolio_value"].transform(lambda v: v / v.cummax() - 1)
        fig = px.line(dd, x="date", y="Drawdown", color="Strategy", title="Drawdown from previous peak",
                      labels={"date": ""})
        fig.update_layout(height=340, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(pct_axis(fig, x=False), width="stretch")


# ---------------------------------------------------------------------------
# Tab 4 – Risk Analysis
# ---------------------------------------------------------------------------

with tab_risk:
    left, right = st.columns(2)
    with left:
        sectors = (pd.Series(w, index=stocks.loc[tickers, "sector"].values)
                   .groupby(level=0).sum().sort_values())
        fig = px.bar(x=sectors.values, y=sectors.index, orientation="h", title="Sector exposure",
                     labels={"x": "Weight", "y": ""}, color_discrete_sequence=[ACCENT])
        st.plotly_chart(pct_axis(fig, y=False), width="stretch")
        st.caption("How your money is spread across parts of the economy.")
    with right:
        comp = pd.DataFrame({"Ticker": tickers, "Weight": w, "Risk contribution": rc})
        comp = comp[comp["Weight"] > 0.001].sort_values("Weight", ascending=False)
        comp = comp.melt(id_vars="Ticker", var_name="Measure", value_name="Share")
        fig = px.bar(comp, x="Ticker", y="Share", color="Measure", barmode="group",
                     title="Share of money vs share of risk",
                     color_discrete_map={"Weight": ACCENT, "Risk contribution": ORANGE})
        st.plotly_chart(pct_axis(fig, x=False), width="stretch")
        st.caption("When the orange bar is taller than the blue one, the stock adds more risk "
                   "than its share of the money.")

    corr = returns.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation of daily returns", aspect="auto")
    fig.update_layout(height=640)
    st.plotly_chart(fig, width="stretch")
    st.caption("Red: the two stocks tend to move together. Blue: they tend to move apart. "
               "Pairs that don't move together are what make diversification work.")


# ---------------------------------------------------------------------------
# Tab 5 – Methodology
# ---------------------------------------------------------------------------

with tab_method:
    st.markdown("""
### What this tool does
Given a set of stocks, it decides **how to split money between them** for the best return per unit of risk,
and tests which allocation method actually works out of sample. It does not predict prices.

### Pipeline
1. **Data** – daily adjusted closing prices (yfinance) and the 3-month T-bill rate (FRED), stored in SQLite and updated incrementally.
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
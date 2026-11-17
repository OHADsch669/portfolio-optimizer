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

import backtest as bt
import optimizer as opt
import pipeline
import stats
import theme
import universe

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Colors: black and white theme (see .streamlit/config.toml and src/theme.py)
# ---------------------------------------------------------------------------

theme.register_plotly_template()
ACCENT, FRONTIER, HIGHLIGHT, MUTED = theme.INK, theme.FRONTIER, theme.GOLD, theme.MUTED

MIN_STOCKS, MAX_STOCKS = 2, 20
DEFAULT_PICK = ["AAPL", "MSFT", "JPM", "JNJ", "PG", "XOM", "NEE", "CAT", "HD", "LIN"]

STRATEGIES = {
    "Max Sharpe ratio": "max_sharpe",
    "Minimum variance": "min_variance",
    "Risk parity": "risk_parity",
    "Equal weight (1/N)": "equal_weight",
}
# One short line under each strategy in the sidebar, written for someone who has never invested
STRATEGY_CAPTIONS = {
    "Max Sharpe ratio": "Aims for the most return for each unit of risk. Looks best on paper.",
    "Minimum variance": "Aims for the smoothest ride, with the smallest ups and downs.",
    "Risk parity": "Every stock is allowed to add the same amount of risk.",
    "Equal weight (1/N)": "The same amount in every stock. Simple, and hard to beat.",
}
DEFAULT_AMOUNT = 10_000

BACKTEST_COLORS = {
    "Equal weight": theme.GREEN, "Min variance": theme.GOLD, "Max Sharpe": theme.INK,
    "Risk parity": theme.PURPLE, "SPY (benchmark)": theme.MUTED,
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
    "port_vol": "Volatility: how much the portfolio's value swings in a typical year. "
                "Lower means a smoother ride.",
    "sharpe": "Return earned for each unit of risk, above what a risk-free government bill pays. "
              "Above 1 is very good.",
    "max_dd": "The biggest drop from a peak to a low: the worst loss you would have felt along the way.",
    "cagr": "The average yearly growth that actually happened, rebalancing every month "
            "using only information available at the time.",
    "bt_vol": "How much the strategy's value swung per year in the test.",
    "last_price": "The most recent closing price, adjusted for dividends and splits.",
    "period_return": "Total gain or loss over the period shown in the chart.",
    "amount": "How much of your money goes into this stock, based on the amount you entered.",
    "vs_spy": " The small line underneath compares it with the S&P 500 over the same period "
              "(pts = percentage points).",
    "stocks_used": "How many of your stocks actually got money. Some strategies leave stocks out.",
    "biggest": "The stock that got the largest share of the money, and how much.",
}

GLOSSARY = {
    "Portfolio": "All the investments you hold together. Here: a set of stocks and how much money goes into each.",
    "Diversification": "Spreading money across stocks that don't all move together, so one bad stock hurts less.",
    "Return": "How much the investment gained (or lost), usually shown per year.",
    "Volatility": "How much the value jumps up and down. A volatility of 15% means that in a typical year "
                  "the result can easily be 15% better or worse than the average.",
    "Max drawdown": "The worst fall from a peak to a low. It shows the biggest loss you would have had to sit through.",
    "Sharpe ratio": "Return per unit of risk, above what a safe government bill pays. "
                    "Below 0.5 is weak, 0.5 to 1 is decent, and above 1 is very good.",
    "Efficient frontier": "The line of the best possible mixes: for each level of risk, the highest return.",
    "Backtest": "A test that replays the past month by month, letting the strategy see only the data it would "
                "have had at the time. It is a fairer test than looking back with hindsight.",
    "Rebalancing": "Going back to the target split from time to time by selling some winners and buying some laggards.",
    "SPY / S&P 500": "A fund that tracks the 500 largest US companies. It is the usual yardstick: "
                     "if a strategy can't beat SPY, you might as well just buy SPY.",
    "CAGR": "Compound annual growth rate: the steady yearly growth that would turn the starting amount "
            "into the final amount.",
}


def usd(x: float) -> str:
    return f"${x:,.0f}"


def sharpe_verdict(s: float) -> str:
    if s < 0:
        return "negative: a safe government bill would have done better"
    if s < 0.5:
        return "weak"
    if s < 1:
        return "decent"
    if s < 2:
        return "very good"
    return "exceptionally high, which in real life rarely lasts (see the Backtest tab)"

# ---------------------------------------------------------------------------
# Look and feel: navy background, card-style metrics, fade-in on tab switch
# ---------------------------------------------------------------------------

st.markdown(theme.CSS, unsafe_allow_html=True)


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


@st.cache_data(ttl=3600)
def last_two_closes(tickers: tuple) -> pd.DataFrame:
    """Last two closing prices per ticker, for the price strip. Columns: last, prev, change, date."""
    placeholders = ",".join("?" * len(tickers))
    query = f"""SELECT ticker, date, adj_close FROM (
                    SELECT ticker, date, adj_close,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                    FROM prices WHERE ticker IN ({placeholders}))
                WHERE rn <= 2"""
    with connect() as conn:
        df = pd.read_sql_query(query, conn, params=list(tickers))
    rows = []
    for t, g in df.sort_values("date").groupby("ticker"):
        if len(g) == 2:
            prev, last = g["adj_close"].values
            rows.append({"ticker": t, "last": last, "prev": prev, "change": last / prev - 1,
                         "date": g["date"].iloc[-1]})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def price_strip(tickers: list[str]) -> None:
    """A moving strip of last prices and daily changes (green up, red down), like a trading-floor screen."""
    order = ["SPY"] + [t for t in tickers if t != "SPY"]
    snap = last_two_closes(tuple(order))
    if snap.empty:
        return
    items = []
    for t in order:
        if t not in snap.index:
            continue
        row = snap.loc[t]
        up = row["change"] >= 0
        label = "S&amp;P 500" if t == "SPY" else t
        items.append(f'<span class="tape-item"><b>{label}</b> ${row["last"]:,.2f} '
                     f'<span class="{"tape-up" if up else "tape-down"}">{"▲" if up else "▼"} '
                     f'{row["change"]:+.2%}</span></span>')
    day = date.fromisoformat(snap["date"].max())
    items_html = "".join(items)
    seconds = max(25, 5 * len(items))                     # same reading speed for long and short lists
    st.markdown(
        f'<div class="tape" title="Last close: {day:%b %d, %Y} · hover to pause">'
        f'<span class="tape-date">Last close {day:%b %d}</span>'
        f'<div class="tape-window"><div class="tape-track" style="animation-duration:{seconds}s">'
        f'{items_html}{items_html}</div></div></div>',
        unsafe_allow_html=True,
    )


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


@st.cache_data(ttl=24 * 3600, show_spinner="Updating the list of US stocks...")
def refresh_stock_list() -> bool:
    """Download the full list of US stocks once a week (checked at most once a day)."""
    with connect() as conn:
        return universe.refresh_universe(conn)


@st.cache_data
def load_universe() -> pd.Series:
    """ticker -> company name, for every stock the user can pick."""
    with connect() as conn:
        return universe.load_universe(conn)["name"]


@st.cache_data
def first_dates(tickers: tuple) -> pd.Series:
    """First date with a price, per ticker."""
    placeholders = ",".join("?" * len(tickers))
    with connect() as conn:
        rows = conn.execute(f"SELECT ticker, MIN(date) FROM prices WHERE ticker IN ({placeholders}) "
                            f"GROUP BY ticker", tickers).fetchall()
    return pd.Series({t: date.fromisoformat(d) for t, d in rows})


@st.cache_data
def load_backtests() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stored study on the original 30 stocks (created by src/backtest.py)."""
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


@st.cache_data(show_spinner=False)
def run_user_backtest(tickers: tuple, cap: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward backtest of the four strategies on the user's own stocks (same rules as backtest.py)."""
    with connect() as conn:
        returns = stats.daily_returns(stats.load_prices(conn, list(tickers)))
        spy = stats.daily_returns(stats.load_prices(conn, [bt.BENCHMARK]))[bt.BENCHMARK]
        rf = pd.read_sql_query("SELECT date, rate FROM risk_free_rates", conn,
                               parse_dates=["date"], index_col="date")["rate"]
    rf_daily = (rf / 100 / stats.TRADING_DAYS).reindex(returns.index).ffill().fillna(0)

    strategies = {
        "equal_weight": lambda mu, s, r: opt.equal_weight(len(mu)),
        "min_variance": lambda mu, s, r: opt.min_variance(s, cap),
        "max_sharpe": lambda mu, s, r: opt.max_sharpe(mu, s, r, cap),
        "risk_parity": lambda mu, s, r: opt.risk_parity(s),
    }
    series = {name: bt.run_backtest(returns, rf_daily, fn)[0] for name, fn in strategies.items()}
    first_day = next(iter(series.values())).index[0]
    series["benchmark_spy"] = spy.loc[first_day:returns.index[-1]]

    summary = pd.DataFrame([{"strategy": name, **bt.metrics(r, rf_daily)} for name, r in series.items()])
    values = pd.concat([pd.DataFrame({"strategy": name, "date": r.index, "portfolio_value": (1 + r).cumprod().values})
                        for name, r in series.items()])
    return summary, values


@st.dialog("Explore the chart", width="large")
def full_screen(fig: go.Figure) -> None:
    """A big view of the chart with all the tools: zoom into an area, move around, reset."""
    big = go.Figure(fig)
    big.update_layout(height=700, dragmode="zoom")
    st.caption("Tools at the top-right of the chart: 🔍 **Zoom**: drag over an area to zoom in · "
               "✥ **Pan**: drag to move around · ➕➖ zoom · 🏠 back to the start. "
               "Close this window to return to the page.")
    st.plotly_chart(big, width="stretch", theme=None, config=theme.FULL_SCREEN_CONFIG)


def show(fig: go.Figure, name: str) -> None:
    """Draw a chart with the app's own Plotly template (theme=None keeps Streamlit from overriding it),
    plus a button that opens it in full screen with the extra tools."""
    # A fixed key keeps the same chart on screen between reruns, so Plotly can animate the change
    # (bars and points glide to their new place when the strategy or the stocks change).
    fig.update_layout(transition=dict(duration=600, easing="cubic-in-out"))
    st.plotly_chart(fig, width="stretch", theme=None, config=theme.PLOTLY_CONFIG, key=f"chart_{name}")
    if st.button("⛶ Full screen", key=f"fs_{name}", type="tertiary",
                 help="Open a bigger view with zoom and move tools"):
        full_screen(fig)


def axis_help(fig: go.Figure, x: tuple[str, str] | None = None, y: tuple[str, str] | None = None,
              hint: tuple[str, str] | None = None) -> go.Figure:
    """
    Axis titles with an ⓘ that shows an explanation on hover (like the help icons on the metrics).
    x, y, hint: (label, explanation). The y label sits above the axis, the hint at the top-right.
    """
    common = dict(xref="paper", yref="paper", showarrow=False, font=dict(size=14, color=theme.TEXT),
                  hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=theme.INK, font=dict(size=13)))
    if x:
        fig.update_xaxes(title=None)
        fig.add_annotation(x=0.5, y=-0.11, xanchor="center", yanchor="top", text=f"{x[0]} → ⓘ",
                           hovertext=x[1], **common)
        fig.update_layout(margin=dict(b=110))
    if y:
        fig.update_yaxes(title=None)
        fig.add_annotation(x=0, y=1.02, xanchor="left", yanchor="bottom", text=f"↑ {y[0]} ⓘ",
                           hovertext=y[1], **common)
    if hint:
        fig.add_annotation(x=1, y=1.02, xanchor="right", yanchor="bottom", text=f"{hint[0]} ⓘ",
                           hovertext=hint[1], **common)
    return fig


def backtest_bottom_line(summary: pd.DataFrame, values: pd.DataFrame, amount: float) -> None:
    """One plain-language paragraph: which strategy won and what the money would have grown to."""
    final = values.sort_values("date").groupby("strategy")["portfolio_value"].last()
    strategies_only = summary[summary["strategy"] != "benchmark_spy"]
    best = strategies_only.loc[strategies_only["sharpe"].idxmax(), "strategy"]
    start_label = f"{values['date'].min():%B %Y}"
    text = (f"**Bottom line:** **{BACKTEST_NAMES[best]}** gave the best return for the risk taken. "
            f"{usd(amount)} invested in {start_label} would have grown to **{usd(amount * final[best])}**")
    if "benchmark_spy" in final:
        text += f", compared with {usd(amount * final['benchmark_spy'])} in the S&P 500 (SPY)"
    text += "."
    if best != "max_sharpe" and "max_sharpe" in final:
        text += (" Notice that **Max Sharpe**, the strategy that looks best on paper, did not win: "
                 "it relies on guesses about future returns, and those guesses are noisy.")
    with st.container(border=True):
        st.markdown(text)


def render_backtest(summary: pd.DataFrame, values: pd.DataFrame, amount: float) -> None:
    backtest_bottom_line(summary, values, amount)
    values = values.copy()
    values["Strategy"] = values["strategy"].map(BACKTEST_NAMES)
    values["portfolio_value"] = values["portfolio_value"] * amount
    fig = px.line(values, x="date", y="portfolio_value", color="Strategy", title=f"Growth of {usd(amount)}",
                  labels={"date": "", "portfolio_value": "Value ($)"}, color_discrete_map=BACKTEST_COLORS)
    fig.update_traces(hovertemplate="%{x|%b %Y}: $%{y:,.0f}")
    fig.update_yaxes(tickprefix="$")
    fig.update_traces(selector=dict(name="SPY (benchmark)"), line=dict(dash="dot"))
    fig.update_layout(height=440, legend=dict(orientation="h", y=-0.15))
    show(fig, f"growth_{len(values)}")

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

    dd = values.sort_values("date")
    dd["Drawdown"] = dd.groupby("Strategy")["portfolio_value"].transform(lambda v: v / v.cummax() - 1)
    fig = px.line(dd, x="date", y="Drawdown", color="Strategy", title="Drawdown from previous peak",
                  labels={"date": ""}, color_discrete_map=BACKTEST_COLORS)
    fig.update_traces(selector=dict(name="SPY (benchmark)"), line=dict(dash="dot"))
    fig.update_layout(height=340, legend=dict(orientation="h", y=-0.2))
    show(pct_axis(fig, x=False), f"drawdown_{len(values)}")


def pct_axis(fig: go.Figure, x: bool = True, y: bool = True) -> go.Figure:
    if x:
        fig.update_xaxes(tickformat=".1~%")
    if y:
        fig.update_yaxes(tickformat=".1~%")
    return fig


def pct_col(label: str, help_key: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="%.1f%%", help=HELP[help_key])


def text_col(label: str, help_key: str) -> st.column_config.TextColumn:
    return st.column_config.TextColumn(label, help=HELP[help_key])


def add_new_stocks(new: list[str], names: pd.Series) -> list[str]:
    """Download and save stocks that are not in the database yet. Returns the ones that failed."""
    failed = []
    with st.status(f"Downloading price history for {', '.join(new)}...", expanded=False) as status:
        with connect() as conn:
            for t in new:
                try:
                    n = universe.add_stock(conn, t, names.get(t, t))
                    status.write(f"{t}: {n:,} days saved")
                except Exception:
                    failed.append(t)
                    status.write(f"{t}: no data found")
        status.update(label="Price history saved", state="complete")
    for cached in (load_assets, load_universe, first_dates, last_price_date):
        cached.clear()
    return failed


# ---------------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------------

data_is_fresh = refresh_market_data()
refresh_stock_list()
names = load_universe()
max_date = last_price_date()
failed = st.session_state.setdefault("failed_tickers", set())

with st.sidebar:
    st.header("Build your portfolio")
    tickers = st.multiselect(
        f"① Pick {MIN_STOCKS}-{MAX_STOCKS} stocks", options=list(names.index),
        default=[t for t in DEFAULT_PICK if t in names.index], max_selections=MAX_STOCKS,
        format_func=lambda t: f"{t} · {names.get(t, '')}",
        placeholder="Type a ticker or company name",
        help=f"Search any US-listed stock by ticker or name. Pick {MIN_STOCKS} to {MAX_STOCKS} stocks; "
             "the optimizer splits your money between them. New stocks are downloaded once and saved.",
    )

    assets = load_assets()
    new = [t for t in tickers if t not in assets.index and t not in failed]
    if new:
        failed |= set(add_new_stocks(new, names))
        assets = load_assets()
    no_data = [t for t in tickers if t in failed]        # warned about only while still selected
    tickers = [t for t in tickers if t not in failed]

    amount = st.number_input(
        "② How much would you invest? ($)", min_value=100, max_value=10_000_000, value=DEFAULT_AMOUNT,
        step=1_000, help="Used only to show the results in dollars. Nothing is bought or sold.")
    strategy_label = st.radio(
        "③ How should the money be split?", list(STRATEGIES),
        help="Not sure? Start with Equal weight. In our 10-year test it gave the best return for the risk.",
    )
    st.caption(STRATEGY_CAPTIONS[strategy_label])          # only the chosen strategy is explained

    with st.expander("Advanced settings"):
        # one date per row: side by side the fields are too narrow and the date gets cut off
        start = st.date_input("From", date(2023, 1, 1), min_value=date(2015, 1, 2), max_value=max_date,
                              help="Start of the history used to estimate returns and risk.")
        end = st.date_input("To", max_date, min_value=date(2015, 1, 2), max_value=max_date)
        min_cap = int(np.ceil(100 / max(len(tickers), 1) / 5) * 5)       # weights must be able to reach 100%
        min_cap = min(min_cap, 95)
        default_cap = min(100, max(20, 2 * min_cap))   # leaves room to optimize: twice the equal-weight share
        # key depends on the number of stocks: the slider is rebuilt cleanly when its range changes
        cap = st.slider("Max weight per stock", min_cap, 100, default_cap, step=5, format="%d%%",
                        key=f"cap_{len(tickers)}",
                        help="No single stock may get more than this share of the money.") / 100
        use_lw = st.checkbox("Use Ledoit-Wolf covariance", value=True,
                             help="A more stable way to estimate how stocks move together. "
                                  "Reduces the effect of random noise in the data.")
    st.caption(f"Prices updated through {max_date:%b %d, %Y}"
               + (" · refreshed automatically" if data_is_fresh else " · using stored data"))

stocks = assets[assets["asset_type"] == "stock"]

price_strip(tickers)
st.title("Portfolio Optimizer")
st.markdown("Decide **how to split your money between stocks**, and see whether that split "
            "would really have worked in the past.")
st.markdown("**🎓 An educational tool. Not investment advice.**")

with st.expander("👋 New to investing? How to use this tool in 3 steps", expanded=True):
    g1, g2, g3 = st.columns(3)
    g1.markdown("**① Pick stocks**  \nIn the sidebar, type companies you know, like *Apple* or *Coca-Cola*. "
                "Choosing several different companies spreads the risk.")
    g2.markdown("**② Choose how to split the money**  \nEach strategy divides your money differently. "
                "Not sure? Start with *Equal weight*.")
    g3.markdown("**③ Read the result**  \nThe box below explains in plain words where the money goes and how "
                "this mix behaved. Then open **Backtest** for a fairer test.")
    st.caption("Hover over any ⓘ icon or column title for a short explanation. "
               "All the terms are explained in the **How it works** tab.")

if no_data:
    st.warning(f"No price data found for {', '.join(no_data)}, so it was left out. "
               "You can remove it from the list.")
if len(tickers) < MIN_STOCKS:
    st.info(f"Pick at least {MIN_STOCKS} stocks in the sidebar to build a portfolio.")
    st.stop()
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

# All stocks are measured over the same period: it starts when the youngest stock has data
firsts = first_dates(tuple(tickers))
youngest = firsts.idxmax()
if firsts[youngest] > start:
    st.info(f"**{youngest}** only has prices from {firsts[youngest]:%b %d, %Y}, "
            f"so the whole portfolio is measured from that date.")
    start = firsts[youngest]

returns = load_returns(tuple(tickers), str(start), str(end))
if len(returns) < 60:
    st.warning("The common period is shorter than three months. Pick an earlier start "
               "or stocks with a longer history.")
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

# The S&P 500 (SPY) over exactly the same days, to compare against
spy_daily = load_returns(("SPY",), str(start), str(end))["SPY"].reindex(port_daily.index).fillna(0)
spy_equity = (1 + spy_daily).cumprod()
spy_ret = spy_daily.mean() * stats.TRADING_DAYS
spy_vol = spy_daily.std() * np.sqrt(stats.TRADING_DAYS)
spy_sharpe = (spy_ret - rf) / spy_vol
spy_dd = (spy_equity / spy_equity.cummax() - 1).min()

tab_build, tab_explore, tab_bt, tab_risk, tab_method = st.tabs(
    ["Portfolio Builder", "Stock Explorer", "Backtest", "Risk Analysis", "How it works"])


# ---------------------------------------------------------------------------
# Tab 1 – Portfolio Builder
# ---------------------------------------------------------------------------

with tab_build:
    st.subheader(f"{strategy_label} portfolio")
    port_ret = opt.portfolio_return(w, mu)
    port_vol = opt.portfolio_volatility(w, sigma)
    port_sharpe = opt.sharpe_ratio(w, mu, sigma, rf)

    def vs_spy(diff: float, pct: bool = True) -> str:
        return (f"{diff * 100:+.1f} pts" if pct else f"{diff:+.2f}") + " vs S&P 500"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Yearly return", f"{port_ret:.1%}", vs_spy(port_ret - spy_ret),
              help=HELP["port_return"] + HELP["vs_spy"])
    k2.metric("Ups & downs", f"{port_vol:.1%}", vs_spy(port_vol - spy_vol), delta_color="inverse",
              help=HELP["port_vol"] + HELP["vs_spy"] + " Here lower is better, so less than the S&P 500 is green.")
    k3.metric("Sharpe ratio", f"{port_sharpe:.2f}", vs_spy(port_sharpe - spy_sharpe, pct=False),
              help=HELP["sharpe"] + HELP["vs_spy"])
    k4.metric("Worst drop", f"{max_dd:.1%}", vs_spy(max_dd - spy_dd),
              help=HELP["max_dd"] + HELP["vs_spy"] + " A smaller drop than the S&P 500 is green.")

    # --- What the money would have become ------------------------------------------------------
    growth = pd.DataFrame({"Your portfolio": equity * amount, "S&P 500 (SPY)": spy_equity * amount})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=growth.index, y=growth["S&P 500 (SPY)"], mode="lines", name="S&P 500 (SPY)",
                             line=dict(color=MUTED, width=2, dash="dot"),
                             hovertemplate="%{x|%b %d, %Y}<br>S&P 500: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=growth.index, y=growth["Your portfolio"], mode="lines", name="Your portfolio",
                             line=dict(color=ACCENT, width=2.5), fill="tozeroy", fillcolor="rgba(17,17,17,0.05)",
                             hovertemplate="%{x|%b %d, %Y}<br>Your portfolio: $%{y:,.0f}<extra></extra>"))
    for col, color in [("Your portfolio", ACCENT), ("S&P 500 (SPY)", MUTED)]:     # final values at the end
        fig.add_annotation(x=growth.index[-1], y=growth[col].iloc[-1], text=f"<b>{usd(growth[col].iloc[-1])}</b>",
                           showarrow=False, xanchor="left", xshift=6, font=dict(color=color, size=13))
    low = growth.min().min()
    fig.update_yaxes(tickprefix="$", range=[low * 0.9, growth.max().max() * 1.05])
    fig.update_layout(title=f"What your {usd(amount)} would have become", height=420,
                      legend=dict(orientation="h", y=-0.3), margin=dict(r=90))
    axis_help(fig,
              x=("Date", "<b>X axis: time</b><br>The period you picked in the settings."),
              y=("Value of your money",
                 "<b>Y axis: value in dollars</b><br>What your investment would have been worth on each day,"
                 "<br>if you had bought this mix at the start and kept it.<br>Dotted line: the same money in the "
                 "S&P 500, for comparison."))
    show(fig, "growth")
    st.caption("⚠️ Hindsight: the split was chosen using this same period, so the line looks better than real "
               "life would have been. The **Backtest** tab shows a fair test.")

    # --- Stocks the strategy gave no money to: explain it, so it doesn't look like a bug -----------
    # Messages go into one fixed slot: the page keeps the same structure whether or not a message is shown,
    # so the charts below stay on screen and animate instead of being redrawn from scratch.
    notes = st.container()
    left_out = [t for t in tickers if weights_by_ticker[t] <= 0.001]
    if left_out:
        mu_by_ticker = pd.Series(mu, index=tickers)
        vol_by_ticker = pd.Series(stock_vol, index=tickers)
        reasons = []
        for t in left_out:
            if strategy == "max_sharpe" and mu_by_ticker[t] < 0:
                why = f"it **lost about {abs(mu_by_ticker[t]):.0%} a year** in this period"
            elif strategy == "max_sharpe":
                why = (f"it gained about {mu_by_ticker[t]:.0%} a year, which is **too little for the ups and "
                       f"downs it adds** ({vol_by_ticker[t]:.0%} a year)")
            else:
                why = (f"it **jumps around a lot** ({vol_by_ticker[t]:.0%} a year) or moves together with your "
                       "other stocks, so it would only add risk")
            reasons.append(f"- **{t}** ({names.get(t, t)}): {why}.")
        n = len(left_out)
        notes.info(
            f"**{strategy_label} gave no money to {n} of your stocks.** This is the optimizer's decision, not an "
            f"error: the {'stock was' if n == 1 else 'stocks were'} checked, but adding "
            f"{'it' if n == 1 else 'them'} would make this portfolio worse.\n\n"
            + "\n".join(reasons)
            + "\n\nWant every stock in? Switch to **Equal weight** or **Risk parity** in the sidebar.",
            icon="ℹ️",
        )
    only_one_split = cap * len(tickers) <= 1 + 1e-9
    if only_one_split:
        notes.info(f"With {len(tickers)} stocks and a max weight of {cap:.0%} per stock, the only possible split is "
                f"{cap:.0%} in each stock, so every strategy gives the same answer. Raise **Max weight per stock** "
                "in Advanced settings (or add stocks) to let the optimizer choose.")
    if strategy == "risk_parity" and w.max() > cap + 1e-6:
        notes.info("Risk parity sets weights by risk contribution, so the max-weight cap is not applied.")

    left, right = st.columns([3, 2])

    with left:
        frontier_note = st.empty()                      # fixed slot, same reason as `notes` above
        fig = go.Figure()
        try:
            frontier = compute_frontier(mu, sigma, cap)
            if frontier["volatility"].max() - frontier["volatility"].min() > 1e-4:   # a line, not one point
                fig.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["return"], mode="lines",
                                         name="Efficient frontier", line=dict(color=FRONTIER, width=3),
                                         hovertemplate="Best mix at this risk<br>risk %{x:.1%}<br>"
                                                       "return %{y:.1%}<extra></extra>"))
        except RuntimeError:
            frontier_note.caption("The efficient frontier could not be computed for these settings.")
        fig.add_trace(go.Scatter(x=stock_vol, y=mu, mode="markers", name="Single stocks", text=tickers,
                                 marker=dict(color=MUTED, size=8, opacity=0.8),
                                 hovertemplate="%{text}<br>vol %{x:.1%}<br>return %{y:.1%}<extra></extra>"))
        w_mv = opt.min_variance(sigma, cap)
        fig.add_trace(go.Scatter(x=[opt.portfolio_volatility(w_mv, sigma)], y=[opt.portfolio_return(w_mv, mu)],
                                 mode="markers", name="Min variance",
                                 marker=dict(color=HIGHLIGHT, size=13, symbol="square")))
        fig.add_trace(go.Scatter(x=[opt.portfolio_volatility(w, sigma)], y=[opt.portfolio_return(w, mu)],
                                 mode="markers", name="Your portfolio",
                                 marker=dict(color=ACCENT, size=16, line=dict(color="white", width=2))))
        fig.update_layout(title="Efficient frontier", height=480, legend=dict(orientation="h", y=-0.3))
        relation = ("<br><br><b>How the two axes relate:</b> more return usually means more risk.<br>"
                    "The best spots are up and to the left: more return for less risk.")
        axis_help(pct_axis(fig),
                  x=("Risk: yearly ups & downs",
                     "<b>X axis: risk (volatility)</b><br>How much the value jumps up and down in a typical year."
                     "<br>Further right = a bumpier ride." + relation),
                  y=("Average yearly return",
                     "<b>Y axis: return</b><br>The average yearly gain in the period you picked."
                     "<br>Higher = more profit." + relation))
        show(fig, "frontier")
        st.caption("How to read it: **higher** = more return, **further right** = more risk. "
                   "The line shows the best return possible for each level of risk. "
                   "Grey dots are single stocks. Almost all sit below the line, "
                   "which is why mixing stocks (diversification) pays off.")

    with right:
        alloc = weights_by_ticker[weights_by_ticker > 0.001].sort_values()
        fig = px.bar(x=alloc.values, y=alloc.index, orientation="h", title=f"Where your {usd(amount)} goes",
                     labels={"x": "Share of your money", "y": ""}, color_discrete_sequence=[ACCENT])
        fig.update_traces(customdata=alloc.values * amount,
                          hovertemplate="%{y}: %{x:.1%} of your money ($%{customdata:,.0f})<extra></extra>")
        fig.update_layout(height=480)
        axis_help(pct_axis(fig, y=False),
                  x=("Share of your money",
                     "<b>X axis: share of your money</b><br>How much of the total goes into each stock."
                     "<br>All the bars together add up to 100%."),
                  y=("Stock",
                     "<b>Y axis: the stocks</b><br>Each bar is one of the stocks you picked."
                     "<br>Hover over a bar to see the amount in dollars."))
        show(fig, "allocation")

    holdings = pd.DataFrame({
        "Ticker": tickers,
        "Company": stocks.loc[tickers, "name"].values,
        "Sector": stocks.loc[tickers, "sector"].values,
        "Weight %": w * 100,
        "Amount $": [usd(x) for x in w * amount],
        "Exp. return %": mu * 100,
        "Volatility %": stock_vol * 100,
        "Risk contribution %": rc * 100,
    })
    holdings = holdings.sort_values("Weight %", ascending=False)      # all stocks, also those left out (0%)
    st.markdown("#### Holdings detail")
    st.caption("Hover over a column title to see what it means."
               + (" Stocks at 0% were checked and left out by the strategy (see the note above)."
                  if left_out else ""))
    st.dataframe(holdings, hide_index=True, width="stretch", column_config={
        "Ticker": text_col("Ticker", "ticker"),
        "Company": text_col("Company", "company"),
        "Sector": text_col("Sector", "sector"),
        "Weight %": pct_col("Weight %", "weight"),
        "Amount $": st.column_config.TextColumn("Amount $", help=HELP["amount"]),
        "Exp. return %": pct_col("Exp. return %", "exp_return"),
        "Volatility %": pct_col("Volatility %", "volatility"),
        "Risk contribution %": pct_col("Risk contribution %", "risk_contrib"),
    })
    st.download_button("Download CSV", holdings.to_csv(index=False).encode(),
                       file_name=f"portfolio_{strategy}.csv", mime="text/csv")

    # --- Compare the four strategies ---------------------------------------------------------
    st.markdown("#### Compare the 4 strategies")
    st.caption("Same stocks, same period. The only difference is how the money is split. "
               "Your current choice is marked with ◀.")
    rows = []
    for label, key in STRATEGIES.items():
        try:
            ws = compute_weights(key, mu, sigma, rf, cap)
        except RuntimeError:
            continue
        eq = (1 + pd.Series(returns.values @ ws)).cumprod()
        biggest = int(np.argmax(ws))
        rows.append({
            "Strategy": label + ("  ◀" if key == strategy else ""),
            "Avg. yearly return %": opt.portfolio_return(ws, mu) * 100,
            "Ups & downs %": opt.portfolio_volatility(ws, sigma) * 100,
            "Sharpe": opt.sharpe_ratio(ws, mu, sigma, rf),
            "Worst drop %": (eq / eq.cummax() - 1).min() * 100,
            "Stocks used": int((ws > 0.001).sum()),
            "Biggest holding": f"{tickers[biggest]} ({ws[biggest]:.0%})",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Avg. yearly return %": pct_col("Avg. yearly return %", "port_return"),
        "Ups & downs %": pct_col("Ups & downs %", "port_vol"),
        "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f", help=HELP["sharpe"]),
        "Worst drop %": pct_col("Worst drop %", "max_dd"),
        "Stocks used": st.column_config.NumberColumn("Stocks used", help=HELP["stocks_used"]),
        "Biggest holding": st.column_config.TextColumn("Biggest holding", help=HELP["biggest"]),
    })
    with st.expander("What's the difference between the strategies?"):
        st.markdown(
            "- **Max Sharpe** looks at both the past returns *and* the risk, and picks the mix with the most "
            "return per unit of risk. It trusts past returns the most, which is why it often disappoints later.\n"
            "- **Minimum variance** ignores returns completely. It only looks at how much stocks jump and how "
            "they move together, and builds the calmest possible mix.\n"
            "- **Risk parity** makes every stock contribute the same amount of risk, so calm stocks get "
            "more money and wild stocks get less.\n"
            "- **Equal weight** puts the same amount in every stock. No estimates, so nothing to get wrong.\n\n"
            "**Why do Max Sharpe and Minimum variance sometimes look alike?** With a low *max weight per stock* "
            "(for example 20% with 10 stocks), both are forced to spread the money and many stocks hit the "
            "ceiling in both. Raise the cap in **Advanced settings** and they pull apart."
        )

    # --- The result in plain words ---------------------------------------------------------
    ranked = weights_by_ticker[weights_by_ticker > 0.001].sort_values(ascending=False)
    top = ranked.head(3)
    parts = [f"**{usd(amount * wt)}** in {names.get(t, t)} ({t})" for t, wt in top.items()]
    rest = len(ranked) - len(top)
    split = ", ".join(parts) + (f", and the rest across {rest} other stock{'s' if rest > 1 else ''}" if rest else "")
    verb = "gained" if port_ret >= 0 else "lost"
    with st.container(border=True):
        st.markdown(
            f"#### 💡 Your result in plain words\n"
            f"Out of **{usd(amount)}**, this strategy puts {split}.\n\n"
            f"- **Return:** from {start:%b %Y} to {end:%b %Y}, this mix {verb} about "
            f"**{abs(port_ret):.0%} a year** on average.\n"
            f"- **Ups and downs:** in a typical year, the result could easily be about "
            f"**{port_vol:.0%} better or worse** than that average.\n"
            f"- **Worst moment:** at its lowest point, {usd(amount)} temporarily fell to "
            f"**{usd(amount * (1 + max_dd))}** ({max_dd:.0%}).\n"
            f"- **Return for the risk (Sharpe {port_sharpe:.2f}):** {sharpe_verdict(port_sharpe)}."
        )
        st.caption("⚠️ These numbers describe the past, measured on the same data the optimizer learned from, "
                   "so they look better than real life will. The **Backtest** tab shows a fairer test.")


# ---------------------------------------------------------------------------
# Tab 2 – Stock Explorer
# ---------------------------------------------------------------------------

with tab_explore:
    options = list(stocks.index) + ["SPY"]
    choice = st.selectbox(
        "Search a stock", options, index=None,
        placeholder="Type a ticker or company name, e.g. Apple or NVDA",
        format_func=lambda t: f"{t} · {assets.loc[t, 'name']}",
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
            for t, color in [(choice, ACCENT), ("SPY", MUTED)]:
                s = view[t].dropna()
                fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, mode="lines", name=t,
                                         line=dict(color=color, width=2)))
            fig.update_layout(yaxis_title="Growth of 100")
        else:
            fig.add_trace(go.Scatter(x=price.index, y=price, mode="lines", name=choice, fill="tozeroy",
                                     line=dict(color=ACCENT, width=2),
                                     fillcolor="rgba(17,17,17,0.07)",
                                     hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>"))
            fig.update_yaxes(range=[price.min() * 0.95, price.max() * 1.03], tickprefix="$")
        fig.update_layout(height=420, legend=dict(orientation="h", y=-0.15), xaxis_title="")
        show(fig, "explorer")


# ---------------------------------------------------------------------------
# Tab 3 – Backtest
# ---------------------------------------------------------------------------

with tab_bt:
    st.subheader("Walk-forward backtest")
    st.caption("Simulates running each strategy in real time: every month the weights are re-estimated "
               f"from the previous {bt.LOOKBACK_DAYS // 252} years only, with {bt.COST_BPS} bps trading costs. "
               "This shows which strategy would actually have worked, not just which looks best on paper.")
    source = st.segmented_control("Stocks", ["Your stocks", "Original 30-stock study"], default="Your stocks")

    if source == "Original 30-stock study":
        summary, values = load_backtests()
        if summary.empty:
            st.info("No stored results yet. Run `python src/backtest.py` once, then refresh.")
        else:
            row = summary.iloc[0]
            st.caption(f"{row.start_date} to {row.end_date} · original 30 stocks · "
                       f"cap {row.max_weight:.0%} per stock.")
            render_backtest(summary, values, amount)
    else:
        common_years = (max_date - firsts.max()).days / 365.25
        needed_years = bt.LOOKBACK_DAYS / 252 + 0.5
        if common_years < needed_years:
            st.info(f"The backtest needs at least {needed_years:.1f} years of history shared by all your stocks; "
                    f"yours have {common_years:.1f} (limited by {youngest}). "
                    "Remove the youngest stock or look at the original 30-stock study.")
        else:
            key = (tuple(sorted(tickers)), cap)
            if st.session_state.get("bt_key") != key:
                st.caption(f"{len(tickers)} stocks · cap {cap:.0%} per stock · "
                           f"about {common_years:.0f} years of shared history.")
                if st.button("Run backtest on my stocks", type="primary"):
                    st.session_state["bt_key"] = key
                    st.rerun()
            else:
                with st.spinner("Running the backtest (about 20-40 seconds the first time)..."):
                    summary, values = run_user_backtest(key[0], cap)
                st.caption(f"{values['date'].min():%b %Y} to {values['date'].max():%b %Y} · "
                           f"{len(tickers)} stocks · cap {cap:.0%} per stock.")
                render_backtest(summary, values, amount)


# ---------------------------------------------------------------------------
# Tab 4 – Risk Analysis
# ---------------------------------------------------------------------------

with tab_risk:
    left, right = st.columns(2)
    with left:
        sectors = (pd.Series(w, index=stocks.loc[tickers, "sector"].values)
                   .groupby(level=0).sum().sort_values())
        sectors = sectors[sectors > 0.001]               # hide sectors with no money in them
        fig = px.bar(x=sectors.values, y=sectors.index, orientation="h", title="Sector exposure",
                     labels={"x": "Weight", "y": ""}, color_discrete_sequence=[ACCENT])
        fig.update_traces(hovertemplate="%{y}: %{x:.1%} of your money<extra></extra>")
        axis_help(pct_axis(fig, y=False),
                  x=("Share of your money",
                     "<b>X axis: share of your money</b><br>How much of the total goes into each part of "
                     "the economy.<br>All the bars together add up to 100%."),
                  y=("Sector",
                     "<b>Y axis: sectors</b><br>The part of the economy each company belongs to "
                     "(technology, energy, health...).<br>A lot of money in one sector means one piece of "
                     "bad news<br>for that sector can hit most of your portfolio."))
        show(fig, "sectors")
        st.caption("How your money is spread across parts of the economy.")
    with right:
        comp = pd.DataFrame({"Ticker": tickers, "Weight": w, "Risk contribution": rc})
        comp = comp[comp["Weight"] > 0.001].sort_values("Weight", ascending=False)
        comp = comp.melt(id_vars="Ticker", var_name="Measure", value_name="Share")
        fig = px.bar(comp, x="Ticker", y="Share", color="Measure", barmode="group",
                     title="Share of money vs share of risk",
                     color_discrete_map={"Weight": ACCENT, "Risk contribution": HIGHLIGHT})
        fig.update_layout(legend=dict(orientation="h", y=-0.3, title=None))
        axis_help(pct_axis(fig, x=False),
                  x=("Stock",
                     "<b>X axis: your stocks</b><br>Each pair of bars is one stock in your portfolio."),
                  y=("Share of the portfolio",
                     "<b>Y axis: share of the portfolio</b><br>Black bar: the share of your money in the stock."
                     "<br>Gold bar: the share of the portfolio's ups and downs that comes from it."
                     "<br>Gold taller than black = the stock adds more risk than money."))
        show(fig, "risk_vs_money")
        st.caption("When the gold bar is taller than the black one, the stock adds more risk "
                   "than its share of the money.")

    corr = returns.corr()
    fig = px.imshow(corr, color_continuous_scale=theme.DIVERGING, zmin=-1, zmax=1,
                    title="Correlation of daily returns", aspect="auto",
                    text_auto=".2f" if len(tickers) <= 10 else False)
    fig.update_layout(height=max(400, 30 * len(tickers) + 180), xaxis_title=None, yaxis_title=None)
    fig.update_traces(hovertemplate="%{y} and %{x}: %{z:.2f}<extra></extra>")
    axis_help(fig, hint=("How to read this",
                         "<b>Each square compares two stocks:</b> the one in its row and the one in its column."
                         "<br>1 = they always move together · 0 = no relation · −1 = they move in opposite "
                         "directions.<br>Red = move together, green = move apart."
                         "<br>More pale and green squares = a better diversified portfolio."))
    show(fig, "correlation")
    st.caption("Red: the two stocks tend to move together. Green: they tend to move apart. "
               "Pairs that don't move together are what make diversification work.")


# ---------------------------------------------------------------------------
# Tab 5 – Methodology
# ---------------------------------------------------------------------------

with tab_method:
    st.markdown("""
### What this tool does
You pick 2 to 20 stocks from any US exchange. The tool decides **how to split money between them** for the best return per unit of risk,
and tests which allocation method actually works out of sample. It does not predict prices.

### Pipeline
1. **Data** – daily adjusted closing prices (yfinance) and the 3-month T-bill rate (FRED), stored in SQLite and updated incrementally.
   The list of pickable stocks comes from the NASDAQ Trader symbol files; a new stock's full history is downloaded once and saved.
   All stocks are measured over the same period, starting when the youngest one has data.
2. **Estimates** – expected returns (annualized historical mean) and the covariance matrix (sample or Ledoit-Wolf shrinkage).
3. **Optimization** – weights sum to 1, no short selling, optional cap per stock; solved with SLSQP (SciPy).
4. **Backtest** – walk-forward: every month, weights are re-estimated from the previous two years only, with transaction costs.

### Strategies
- **Max Sharpe** – highest excess return per unit of risk. Most sensitive to errors in expected returns.
- **Minimum variance** – lowest possible risk; ignores expected returns.
- **Risk parity** – every stock contributes the same share of total risk.
- **Equal weight (1/N)** – the naive benchmark that is notoriously hard to beat.

### Limitations
- **Survivorship bias:** you pick stocks that exist today, so companies that failed along the way are missing, which flatters past results.
- **Expected returns are noisy:** historical averages are weak forecasts; see how Max Sharpe performs out of sample.
- Results are for analysis and education only and are **not investment advice**.
""")
    st.markdown("### Glossary for beginners")
    for term, meaning in GLOSSARY.items():
        st.markdown(f"**{term}**: {meaning}")
"""
Portfolio optimizer: given expected returns (mu) and a covariance matrix
(sigma), find the weights w for each strategy.

Every strategy solves a constrained optimization problem:
    decision variables : w_i  (share of the portfolio in stock i)
    constraints        : sum(w) = 1,  0 <= w_i <= max_weight  (long-only, capped)
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import stats


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """Expected annual return of the portfolio: w' * mu"""
    return float(w @ mu)


def portfolio_volatility(w: np.ndarray, sigma: np.ndarray) -> float:
    """Annual volatility of the portfolio: sqrt(w' * Sigma * w)"""
    return float(np.sqrt(w @ sigma @ w))


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    """Excess return per unit of risk."""
    return (portfolio_return(w, mu) - rf) / portfolio_volatility(w, sigma)


# ---------------------------------------------------------------------------
# Generic constrained solver
# ---------------------------------------------------------------------------

def _solve(objective, n: int, max_weight: float, extra_constraints: list | None = None) -> np.ndarray:
    """Minimize `objective(w)` subject to sum(w)=1 and 0 <= w_i <= max_weight."""
    if max_weight * n < 1:
        raise ValueError(f"max_weight={max_weight} is too small for {n} stocks (weights can't sum to 1)")

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    constraints += extra_constraints or []
    bounds = [(0.0, max_weight)] * n
    w0 = np.full(n, 1 / n)  # start from the equal-weight portfolio

    result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    w = result.x
    w[w < 1e-6] = 0        # clean numerical dust
    return w / w.sum()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def equal_weight(n: int) -> np.ndarray:
    """1/N: the naive benchmark that is surprisingly hard to beat."""
    return np.full(n, 1 / n)


def min_variance(sigma: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    """Lowest possible risk. Ignores expected returns entirely, so it is robust to their noise."""
    return _solve(lambda w: w @ sigma @ w, len(sigma), max_weight)


def max_sharpe(mu: np.ndarray, sigma: np.ndarray, rf: float, max_weight: float = 1.0) -> np.ndarray:
    """Best return per unit of risk (maximize Sharpe = minimize -Sharpe)."""
    return _solve(lambda w: -sharpe_ratio(w, mu, sigma, rf), len(mu), max_weight)


def risk_parity(sigma: np.ndarray) -> np.ndarray:
    """
    Each stock contributes the same share of total risk.
    Solved via the convex form: minimize 0.5*w'Sw - (1/n)*sum(log w), then rescale.
    """
    n = len(sigma)
    objective = lambda w: 0.5 * w @ sigma @ w - np.sum(np.log(w)) / n
    result = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                      bounds=[(1e-8, None)] * n, options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x / result.x.sum()


def risk_contributions(w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Share of total portfolio risk coming from each stock (sums to 1)."""
    marginal = sigma @ w
    return w * marginal / (w @ sigma @ w)


def efficient_frontier(mu: np.ndarray, sigma: np.ndarray, max_weight: float = 1.0,
                       points: int = 30) -> pd.DataFrame:
    """Minimum-risk portfolio for a range of target returns."""
    n = len(mu)
    r_min = portfolio_return(min_variance(sigma, max_weight), mu)
    r_max = portfolio_return(_solve(lambda w: -(w @ mu), n, max_weight), mu)

    rows = []
    for target in np.linspace(r_min, r_max, points):
        hit_target = {"type": "eq", "fun": lambda w, t=target: w @ mu - t}
        w = _solve(lambda w: w @ sigma @ w, n, max_weight, [hit_target])
        rows.append({"return": portfolio_return(w, mu), "volatility": portfolio_volatility(w, sigma)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compare all strategies – run this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    START = "2023-01-01"
    MAX_WEIGHT = 0.20

    with sqlite3.connect(stats.DB_PATH) as conn:
        tickers = stats.get_stock_tickers(conn)
        rets = stats.daily_returns(stats.load_prices(conn, tickers, start=START))
        rf = stats.risk_free_rate(conn, START)

    mu = stats.expected_returns(rets).values
    sigma = stats.covariance_matrix(rets, method="ledoit_wolf").values

    portfolios = {
        "Equal weight": equal_weight(len(mu)),
        "Min variance": min_variance(sigma, MAX_WEIGHT),
        "Max Sharpe":   max_sharpe(mu, sigma, rf, MAX_WEIGHT),
        "Risk parity":  risk_parity(sigma),
    }

    summary = pd.DataFrame({
        name: {
            "return": portfolio_return(w, mu),
            "volatility": portfolio_volatility(w, sigma),
            "sharpe": sharpe_ratio(w, mu, sigma, rf),
            "holdings": int((w > 0.001).sum()),
            "largest": w.max(),
        }
        for name, w in portfolios.items()
    }).T
    print(f"In-sample comparison, {START} to today, cap {MAX_WEIGHT:.0%} per stock\n")
    print(summary.round(3).to_string())

    for name in ["Min variance", "Max Sharpe"]:
        w = pd.Series(portfolios[name], index=tickers)
        print(f"\n{name} – top holdings:")
        print(w[w > 0.001].sort_values(ascending=False).map("{:.1%}".format).to_string())

    rc = risk_contributions(portfolios["Risk parity"], sigma)
    print(f"\nRisk parity check – risk contribution per stock: "
          f"min {rc.min():.1%}, max {rc.max():.1%} (target {1/len(rc):.1%})")
"""Out-of-sample comparison of Minimum Variance and HRP portfolios."""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


DEFAULT_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "SBIN.NS", "HINDUNILVR.NS",
]
DEFAULT_START = date(2025, 1, 1)
DEFAULT_TRAIN_END = date(2026, 4, 30)
DEFAULT_TEST_START = date(2026, 5, 1)
DEFAULT_END = date(2026, 10, 31)
TRADING_DAYS = 252


def clean_tickers(raw: str) -> list[str]:
    """Return de-duplicated Yahoo symbols, preserving entered order."""
    result: list[str] = []
    for item in raw.replace("\n", ",").split(","):
        ticker = item.strip().upper()
        if ticker and ticker not in result:
            result.append(ticker)
    return result


@st.cache_data(show_spinner=False, ttl=3600)
def load_close_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    # Yahoo's end date is exclusive; request the next calendar day.
    raw = yf.download(list(tickers), start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1),
                      auto_adjust=False, progress=False, group_by="column", threads=True)
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.reindex(columns=list(tickers))


def minimum_variance_weights(returns: pd.DataFrame) -> pd.Series:
    cov = returns.cov().values * TRADING_DAYS
    n = len(returns.columns)
    objective = lambda w: float(w @ cov @ w)
    result = minimize(objective, np.repeat(1 / n, n), method="SLSQP",
                      bounds=[(0.0, 1.0)] * n,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                      options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        raise ValueError(f"MVP optimizer did not converge: {result.message}")
    return pd.Series(result.x, index=returns.columns, name="MVP")


def correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    return ((1 - corr.clip(-1, 1)) / 2).pow(0.5)


def quasi_diagonalize(link: np.ndarray, labels: list[str]) -> list[str]:
    """Order leaf labels so closely linked assets sit next to each other."""
    n = len(labels)
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    for i, (left, right, *_rest) in enumerate(link):
        clusters[n + i] = clusters[int(left)] + clusters[int(right)]
    return [labels[i] for i in clusters[2 * n - 2]]


def cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    sub_cov = cov.loc[items, items]
    inv_diag = 1 / np.diag(sub_cov.values)
    ivp = inv_diag / inv_diag.sum()
    return float(ivp @ sub_cov.values @ ivp)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    cov, corr = returns.cov(), returns.corr()
    distance = correl_dist(corr)
    condensed = squareform(distance.values, checks=False)
    tree = linkage(condensed, method="single")
    ordered = quasi_diagonalize(tree, list(returns.columns))
    weights = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            left_var, right_var = cluster_variance(cov, left), cluster_variance(cov, right)
            alpha = 1 - left_var / (left_var + right_var)
            weights.loc[left] *= alpha
            weights.loc[right] *= 1 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters
    return weights.reindex(returns.columns).rename("HRP")


def performance(daily_returns: pd.Series, risk_free_rate: float) -> dict[str, float]:
    periods = len(daily_returns)
    wealth = (1 + daily_returns).prod()
    ann_return = wealth ** (TRADING_DAYS / periods) - 1
    ann_vol = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol else np.nan
    running_max = (1 + daily_returns).cumprod().cummax()
    drawdown = (1 + daily_returns).cumprod() / running_max - 1
    return {"Annualized return": ann_return, "Annualized volatility": ann_vol,
            "Sharpe ratio": sharpe, "Maximum drawdown": drawdown.min(),
            "Test observations": periods}


def csv_download(df: pd.DataFrame | pd.Series) -> bytes:
    return df.to_csv().encode("utf-8")


st.set_page_config(page_title="MVP vs HRP | NIFTY 50", layout="wide")
st.title("MVP vs HRP: out-of-sample portfolio comparison")
st.caption("Daily Yahoo Finance **Close** prices only • long-only portfolios • fixed weights during the test period")

with st.sidebar:
    st.header("Inputs")
    ticker_text = st.text_area("Yahoo Finance tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), height=130)
    st.caption("Default is a sensible liquid, large-cap NIFTY 50 basket, not a dynamically verified ‘top 10’ ranking. Edit it freely.")
    risk_free_pct = st.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=25.0, value=0.0, step=0.25)
    run = st.button("Download data and run comparison", type="primary", use_container_width=True)

st.info("Training: 1 Jan 2025–30 Apr 2026. Test: 1 May 2026–31 Oct 2026. Prices are requested through 31 Oct; because 31 Oct 2026 is a Saturday, the final expected trading close is 30 Oct 2026.")

if run:
    tickers = clean_tickers(ticker_text)
    if len(tickers) < 2:
        st.error("Enter at least two valid Yahoo Finance ticker symbols.")
        st.stop()
    with st.spinner("Downloading Close prices from Yahoo Finance..."):
        prices = load_close_prices(tuple(tickers), DEFAULT_START, DEFAULT_END)
    if prices.empty:
        st.error("Yahoo Finance returned no prices. Check your connection and ticker symbols.")
        st.stop()

    missing = prices.columns[prices.isna().all()].tolist()
    partial = prices.columns[prices.isna().any() & ~prices.isna().all()].tolist()
    usable = prices.drop(columns=missing).dropna(axis=1)
    if missing:
        st.warning("No Close data returned for: " + ", ".join(missing) + ". They were excluded.")
    if partial:
        st.warning("Incomplete history for: " + ", ".join(partial) + ". To avoid imputation and leakage, they were excluded.")
    if len(usable.columns) < 2:
        st.error("Fewer than two tickers have complete Close-price history across the requested window.")
        st.stop()

    returned_start, returned_end = usable.index.min().date(), usable.index.max().date()
    st.write(f"**Available complete data:** {returned_start:%d %b %Y} to {returned_end:%d %b %Y}; {len(usable.columns)} of {len(tickers)} requested tickers retained.")
    if returned_start > DEFAULT_START or returned_end < date(2026, 10, 30):
        st.warning("Yahoo Finance did not return the full expected window. Results use only the returned common dates; review the data download below.")

    train_prices = usable.loc[:str(DEFAULT_TRAIN_END)]
    test_prices = usable.loc[str(DEFAULT_TEST_START):]
    if train_prices.empty or test_prices.empty:
        st.error("The returned data do not cover both the training and test periods.")
        st.stop()
    train_returns = train_prices.pct_change().dropna()
    # The first test return must use the prior close (30 Apr) as its denominator.
    boundary_prices = usable.loc[usable.index >= pd.Timestamp(DEFAULT_TRAIN_END)].copy()
    test_returns = boundary_prices.pct_change().loc[str(DEFAULT_TEST_START):].dropna()
    if len(train_returns) < 30 or len(test_returns) < 10:
        st.error("Too few daily observations were returned to form a reliable comparison.")
        st.stop()

    try:
        mvp, hrp = minimum_variance_weights(train_returns), hrp_weights(train_returns)
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.error(str(exc))
        st.stop()

    strategy_returns = pd.DataFrame({"MVP": test_returns @ mvp, "HRP": test_returns @ hrp})
    values = 100 * (1 + strategy_returns).cumprod()
    metrics = pd.DataFrame({name: performance(strategy_returns[name], risk_free_pct / 100) for name in strategy_returns}).T
    allocations = pd.concat([mvp, hrp], axis=1).fillna(0)

    st.subheader("Out-of-sample results")
    st.caption(f"Weights were estimated from {train_returns.index.min():%d %b %Y} through {train_returns.index.max():%d %b %Y}, then held constant for {test_returns.index.min():%d %b %Y} through {test_returns.index.max():%d %b %Y}. Sharpe ratio uses a {risk_free_pct:.2f}% annual risk-free rate.")
    display_metrics = metrics.copy()
    for col in ["Annualized return", "Annualized volatility", "Maximum drawdown"]:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.2%}")
    display_metrics["Sharpe ratio"] = display_metrics["Sharpe ratio"].map(lambda x: f"{x:.3f}")
    st.dataframe(display_metrics, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.line(values, title="Portfolio value (₹100 starting value)", labels={"value": "Value", "index": "Date"}), use_container_width=True)
    with col2:
        risk_return = metrics.reset_index(names="Strategy")
        fig = px.scatter(risk_return, x="Annualized volatility", y="Annualized return", text="Strategy", title="Annualized return vs. risk")
        fig.update_traces(textposition="top center", marker={"size": 14})
        fig.update_xaxes(tickformat=".1%")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fixed allocations from training data")
    left, right = st.columns(2)
    for column, pane in zip(["MVP", "HRP"], [left, right]):
        with pane:
            weights_for_chart = allocations[column][allocations[column] > 0.0001]
            st.plotly_chart(px.pie(values=weights_for_chart.values, names=weights_for_chart.index, title=f"{column} allocation"), use_container_width=True)
    allocation_table = (allocations * 100).round(2)
    allocation_table.columns = ["MVP weight (%)", "HRP weight (%)"]
    st.dataframe(allocation_table, use_container_width=True)

    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Close prices CSV", csv_download(usable), "close_prices.csv", "text/csv")
    d2.download_button("Allocations CSV", csv_download(allocations), "allocations.csv", "text/csv")
    d3.download_button("Test returns CSV", csv_download(strategy_returns), "test_portfolio_returns.csv", "text/csv")
    d4.download_button("Metrics CSV", csv_download(metrics), "performance_metrics.csv", "text/csv")
    with st.expander("Data and methodology notes"):
        st.write("MVP minimizes annualized sample-covariance variance subject to weights summing to 100% and each weight staying between 0% and 100%. HRP uses correlation distance, single-linkage clustering, quasi-diagonalization, and recursive bisection. Neither method uses test-period returns when estimating weights.")
        st.write("Prices with missing values are excluded rather than filled. The test return on 1 May, when present, is calculated from the previous available Close, which preserves the close-to-close boundary return without training on test data.")
else:
    st.write("Choose tickers and click **Download data and run comparison**.")

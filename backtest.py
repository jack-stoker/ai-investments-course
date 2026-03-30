"""
Simple moving-average crossover backtest on SVXY daily price data.
Uses: pandas, numpy, statsmodels, matplotlib, scipy, pyarrow
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "Homework1", "SVXY.csv")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Compute signals: 20-day / 60-day SMA crossover
# ---------------------------------------------------------------------------
SHORT_WIN = 20
LONG_WIN  = 60

df["sma_short"] = df["close"].rolling(SHORT_WIN).mean()
df["sma_long"]  = df["close"].rolling(LONG_WIN).mean()

# Daily log returns
df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

# Signal: 1 (long) when short SMA > long SMA, else 0 (cash)
df["signal"] = np.where(df["sma_short"] > df["sma_long"], 1.0, 0.0)
# Strategy return: previous day's signal applied to today's return
df["strat_ret"] = df["signal"].shift(1) * df["log_ret"]

# Drop warm-up rows
df = df.dropna(subset=["strat_ret", "log_ret"]).copy()

# ---------------------------------------------------------------------------
# 3. Performance metrics
# ---------------------------------------------------------------------------
TRADING_DAYS = 252

def annualised_return(log_rets):
    return log_rets.mean() * TRADING_DAYS

def annualised_vol(log_rets):
    return log_rets.std() * np.sqrt(TRADING_DAYS)

def sharpe(log_rets, rf=0.0):
    excess = log_rets.mean() - rf / TRADING_DAYS
    return (excess / log_rets.std()) * np.sqrt(TRADING_DAYS)

def max_drawdown(log_rets):
    cum = log_rets.cumsum()
    rolling_max = cum.cummax()
    drawdown = cum - rolling_max
    return drawdown.min()

def calmar(log_rets):
    mdd = abs(max_drawdown(log_rets))
    if mdd == 0:
        return np.nan
    return annualised_return(log_rets) / mdd

metrics = {}
for label, series in [("Buy & Hold", df["log_ret"]), ("SMA Strategy", df["strat_ret"])]:
    metrics[label] = {
        "Ann. Return":  f"{annualised_return(series):.2%}",
        "Ann. Vol":     f"{annualised_vol(series):.2%}",
        "Sharpe":       f"{sharpe(series):.3f}",
        "Max Drawdown": f"{max_drawdown(series):.2%}",
        "Calmar":       f"{calmar(series):.3f}",
    }

print("\n=== Backtest Results: SVXY SMA(20/60) Crossover ===\n")
header = f"{'Metric':<16}" + "".join(f"{k:<18}" for k in metrics)
print(header)
print("-" * len(header))
for metric in list(metrics["Buy & Hold"].keys()):
    row = f"{metric:<16}" + "".join(f"{metrics[k][metric]:<18}" for k in metrics)
    print(row)

# ---------------------------------------------------------------------------
# 4. OLS alpha/beta vs buy-and-hold (statsmodels)
# ---------------------------------------------------------------------------
X = add_constant(df["log_ret"])
ols = OLS(df["strat_ret"], X).fit()
alpha_ann = ols.params["const"] * TRADING_DAYS
beta       = ols.params["log_ret"]
t_alpha    = ols.tvalues["const"]
p_alpha    = ols.pvalues["const"]

print(f"\nOLS regression (strategy ~ buy-and-hold):")
print(f"  Annualised alpha: {alpha_ann:.4f}  (t={t_alpha:.2f}, p={p_alpha:.3f})")
print(f"  Beta:             {beta:.4f}")

# ---------------------------------------------------------------------------
# 5. t-test: is strategy mean return significantly different from zero?
# ---------------------------------------------------------------------------
t_stat, p_val = stats.ttest_1samp(df["strat_ret"].dropna(), 0.0)
print(f"\nOne-sample t-test on strategy daily returns:")
print(f"  t-statistic: {t_stat:.4f}   p-value: {p_val:.4f}")

# ---------------------------------------------------------------------------
# 6. Save results to Parquet (uses pyarrow)
# ---------------------------------------------------------------------------
out_cols = ["date", "close", "sma_short", "sma_long",
            "signal", "log_ret", "strat_ret"]
out_path = os.path.join(os.path.dirname(__file__), "backtest_results.parquet")
df[out_cols].to_parquet(out_path, index=False)
print(f"\nResults saved to: {out_path}")

# ---------------------------------------------------------------------------
# 7. Plot cumulative returns
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

cum_bh     = df["log_ret"].cumsum()
cum_strat  = df["strat_ret"].cumsum()

axes[0].plot(df["date"], cum_bh,    label="Buy & Hold",    color="steelblue")
axes[0].plot(df["date"], cum_strat, label="SMA Strategy",  color="darkorange")
axes[0].set_ylabel("Cumulative Log Return")
axes[0].set_title("SVXY: SMA(20/60) Crossover vs Buy & Hold")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Drawdown
dd_bh    = cum_bh    - cum_bh.cummax()
dd_strat = cum_strat - cum_strat.cummax()
axes[1].fill_between(df["date"], dd_bh,    color="steelblue",  alpha=0.4, label="Buy & Hold")
axes[1].fill_between(df["date"], dd_strat, color="darkorange", alpha=0.4, label="SMA Strategy")
axes[1].set_ylabel("Drawdown")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "backtest_plot.png")
fig.savefig(plot_path, dpi=150)
print(f"Plot saved to:    {plot_path}\n")

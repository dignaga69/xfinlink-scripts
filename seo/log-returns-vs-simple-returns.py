# Full write-up: https://xfinlink.com/blog/log-returns-vs-simple-returns
#
# Log returns vs simple returns: log returns add across time, simple returns
# average across holdings. Ten years of split-adjusted closes, six companies.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["NVDA", "AAPL", "MSFT", "XOM", "JNJ", "KO"]
px = xfl.prices(tickers, start="2015-01-01", end="2024-12-31", fields=["adj_close"])

# 1. Across time: adding simple returns is not compounding them.
rows = []
for t in tickers:
    s = px[px["ticker"] == t].sort_values("date")["adj_close"]
    simple = s.pct_change().dropna()
    log = np.log1p(simple)
    rows.append({
        "ticker": t,
        "added_up_pct": 100 * simple.sum(),
        "compounded_pct": 100 * ((1 + simple).prod() - 1),
        "exp_sum_log_pct": 100 * np.expm1(log.sum()),
        "ann_vol_pct": 100 * simple.std() * np.sqrt(252),
        "mean_simple_ann_pct": 100 * simple.mean() * 252,
        "mean_log_ann_pct": 100 * log.mean() * 252,
        "half_variance_ann_pct": 100 * 0.5 * simple.var() * 252,
        "cagr_pct": 100 * np.expm1(log.mean() * 252),
    })
d = pd.DataFrame(rows)
d["gap_pct"] = d["mean_simple_ann_pct"] - d["mean_log_ann_pct"]
pd.set_option("display.width", 250)
print(d.round(2).to_string(index=False))

# 2. Across holdings: a portfolio earns the average simple return, not the
#    average log return.
wide = px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
r = wide.pct_change().dropna()
port_simple = r[["NVDA", "KO"]].mean(axis=1)
port_from_logs = np.expm1(np.log1p(r[["NVDA", "KO"]]).mean(axis=1))
gap = (port_simple - port_from_logs).abs()
print("\nequal-weight NVDA/KO, rebalanced daily")
print("mean absolute daily gap (pct pts):", round(100 * gap.mean(), 4))
print("compounded, simple space:  ", round(100 * ((1 + port_simple).prod() - 1), 1))
print("compounded, averaged logs: ", round(100 * ((1 + port_from_logs).prod() - 1), 1))
day = gap.idxmax()
print("widest day", day.date(),
      "NVDA", round(100 * r.loc[day, "NVDA"], 2),
      "KO", round(100 * r.loc[day, "KO"], 2),
      "simple avg", round(100 * port_simple.loc[day], 2),
      "log avg", round(100 * port_from_logs.loc[day], 2))

# 3. Chart: one NVDA series, compounded against added up.
s = px[px["ticker"] == "NVDA"].sort_values("date")
step = s["adj_close"].pct_change().fillna(0)
compounded = (1 + step).cumprod()
added_up = 1 + step.cumsum()

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")
ax.plot(s["date"], compounded, color="#3b82f6", linewidth=2,
        label="Compounded: (1+r) multiplied")
ax.plot(s["date"], added_up, color="#9ca3af", linewidth=2,
        label="Added up: 1 + sum of r")
ax.set_yscale("log")
ax.set_ylabel("Value of $1, log scale", color="#e0e0e0")
ax.set_title("Adding daily returns is not compounding them: NVDA, 2015-2024",
             color="#e0e0e0", fontsize=13)
ax.tick_params(colors="#9ca3af")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color("#333333")
ax.grid(axis="y", color="#1f1f1f", linewidth=0.8)
ax.set_axisbelow(True)
ax.annotate(f"${compounded.iloc[-1]:,.0f}", xy=(s['date'].iloc[-1], compounded.iloc[-1]),
            xytext=(-6, 8), textcoords="offset points", color="#3b82f6",
            fontsize=11, ha="right")
ax.annotate(f"${added_up.iloc[-1]:,.2f}", xy=(s['date'].iloc[-1], added_up.iloc[-1]),
            xytext=(-6, -16), textcoords="offset points", color="#9ca3af",
            fontsize=11, ha="right")
leg = ax.legend(frameon=False, loc="upper left")
for t in leg.get_texts():
    t.set_color("#e0e0e0")
plt.tight_layout()
plt.savefig("log-returns-vs-simple-returns.png", dpi=150, facecolor="#0a0a0a")

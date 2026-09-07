# Full write-up: https://xfinlink.com/blog/volatility-shock-half-life-python
"""How long does a volatility spike take to fade?

Fits an AR(1) to log realised volatility at four sampling horizons and compares
the implied half-life with the decay actually observed after real spikes.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2016-09-01", "2026-09-04"
names = [t for t in sorted(xfl.index("sp500")["ticker"].unique()) if "-" not in t]
names += ["SPY"]


def grab(t):
    for a in range(4):
        try:
            return xfl.prices(t, start=START, end=END, fields=["adj_close"])
        except Exception:
            if a == 3:
                return pd.DataFrame()
            time.sleep(2 * (a + 1))
    return pd.DataFrame()


with ThreadPoolExecutor(4) as ex:
    px = pd.concat([d for d in ex.map(grab, names) if len(d)], ignore_index=True)

P = (px[px["ticker"].isin(names)].drop_duplicates(["ticker", "date"])
       .pivot(index="date", columns="ticker", values="adj_close").sort_index())
P = P.reindex(P["SPY"].dropna().index)
R = P.pct_change().iloc[1:]
R = R.loc[:, R.notna().all()]
R = R.drop(columns=R.columns[((R > 1.0) | (R < -0.5)).any()])  # corporate-action artefacts
S = R.drop(columns="SPY")


def rvol(X, h):
    """Annualised realised volatility over non-overlapping h-day blocks."""
    nb = len(X) // h
    return np.sqrt((252.0 / h) * (X.iloc[:nb * h] ** 2)
                   .groupby(np.repeat(np.arange(nb), h)).sum())


def pooled_phi(L):
    """AR(1) slope on log realised volatility, pooled with one mean per name."""
    D = L - L.mean()
    x0, x1 = D.iloc[:-1].values.ravel(), D.iloc[1:].values.ravel()
    return float((x0 @ x1) / (x0 @ x0))


half = lambda phi, h: h * np.log(0.5) / np.log(phi)

print(f"Sample: {R.shape[1] - 1} S&P 500 stocks plus SPY, {len(R)} trading days, "
      f"{R.index[0].date()} to {R.index[-1].date()}\n")
print("AR(1) HALF-LIFE OF LOG REALISED VOLATILITY, BY SAMPLING HORIZON")
print(f"{'block':>7}  {'obs/name':>8}  {'phi (stocks)':>12}  {'half-life':>10}  "
      f"{'phi (SPY)':>9}  {'half-life':>10}")
hl_s, hl_i, hs = [], [], [5, 10, 21, 42]
for h in hs:
    ps, pi = pooled_phi(np.log(rvol(S, h))), pooled_phi(np.log(rvol(R[["SPY"]], h)))
    hl_s.append(half(ps, h))
    hl_i.append(half(pi, h))
    print(f"{h:>5}d  {len(S) // h:>8}  {ps:>12.4f}  {hl_s[-1]:>7.1f}d  "
          f"{pi:>9.4f}  {hl_i[-1]:>7.1f}d")

# --- what actually happens after a spike -------------------------------------
LW = np.log(rvol(S, 5))
phi_w = pooled_phi(LW)
mu = LW.mean()
spy_w = np.log(rvol(R[["SPY"]], 5))["SPY"].values
calm = spy_w < np.quantile(spy_w, 0.95)

K = [0, 1, 2, 4, 8, 13, 26, 52]


def decay(mask):
    act = {k: [] for k in K}
    mod = {k: [] for k in K}
    n = 0
    for c in LW.columns:
        y = LW[c].values
        hit = np.where((y >= np.quantile(y, 0.95)) & mask)[0]
        hit = hit[hit + max(K) < len(y)]
        n += len(hit)
        for t in hit:
            for k in K:
                act[k].append(np.exp(y[t + k] - mu[c]))
                mod[k].append(np.exp(phi_w ** k * (y[t] - mu[c])))
    return n, {k: float(np.median(act[k])) for k in K}, {k: float(np.median(mod[k])) for k in K}


n_all, act, mod = decay(np.ones(len(LW), bool))
n_idio, act_i, _ = decay(calm)
s0 = np.log(act[0])

print(f"\nDECAY AFTER A TOP-5% VOLATILITY WEEK ({n_all} stock-weeks)")
print(f"{'weeks after':>11}  {'actual':>8}  {'AR(1)':>8}  {'spike left':>10}  {'ex-market':>10}")
for k in K:
    print(f"{k:>11}  {act[k]:>7.2f}x  {mod[k]:>7.2f}x  "
          f"{100 * np.log(act[k]) / s0:>9.1f}%  {act_i[k]:>9.2f}x")
print(f"Excluding the 5% of weeks when SPY itself spiked: {n_idio} stock-weeks.")

print("\nSHARE OF A SPIKE AN OPTION OF N WEEKS SHOULD PRICE")
print(f"{'maturity':>9}  {'actual':>7}  {'vol x':>6}  {'AR(1)':>7}  {'vol x':>6}")
frac = [np.log(act[k]) / s0 for k in K]
for N in [1, 4, 13, 26, 52]:
    ks = np.arange(1, N + 1)
    e = float(np.interp(ks, K, frac).mean())
    m = float(np.mean(phi_w ** ks))
    print(f"{N:>8}w  {e:>7.3f}  {2 ** e:>5.2f}x  {m:>7.3f}  {2 ** m:>5.2f}x")

# --- chart -------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 10})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))

a1.plot(hs, hl_s, "o-", color="#3b82f6", label="S&P 500 stocks")
a1.plot(hs, hl_i, "s-", color="#f59e0b", label="SPY")
a1.axhline(hl_s[0], ls="--", lw=1, color="#9ca3af",
           label="what a single AR(1) implies")
a1.set_xlabel("Length of the block used to measure volatility (trading days)")
a1.set_ylabel("Estimated half-life (trading days)")
a1.set_title("The half-life depends on how you measure it")
a1.legend(frameon=False, fontsize=8)

a2.plot(K, [act[k] for k in K], "o-", color="#3b82f6", label="what actually happens")
a2.plot(K, [mod[k] for k in K], "--", color="#f59e0b", label="AR(1) forecast")
a2.plot(K, [act_i[k] for k in K], ":", color="#9ca3af",
        label="excluding market-wide stress")
a2.axhline(1.0, lw=0.8, color="#3f3f3f")
a2.set_yscale("log")
a2.set_yticks([1.0, 1.25, 1.5, 2.0, 3.0])
a2.set_yticklabels(["1.0x", "1.25x", "1.5x", "2.0x", "3.0x"])
a2.set_xlabel("Weeks after the spike")
a2.set_ylabel("Volatility relative to the stock's own average")
a2.set_title("Volatility fades far more slowly than AR(1) predicts")
a2.legend(frameon=False, fontsize=8)

for ax in (a1, a2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("volatility-shock-half-life-python.png", dpi=150,
            facecolor="#0a0a0a")

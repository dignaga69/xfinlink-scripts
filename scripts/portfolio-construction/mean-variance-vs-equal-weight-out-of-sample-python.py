# Full write-up: https://xfinlink.com/blog/mean-variance-vs-equal-weight-out-of-sample-python
#
# Does mean-variance optimisation beat equal weighting out of sample?
# Nineteen annual rebalances on point-in-time S&P 500 rosters, 2006-2024.

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy.optimize import minimize

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(300)

YEARS = list(range(2006, 2025))
SIZES = [10, 25, 50, 100]
DRAWS = 20
CACHE = "."
IMG = "mean-variance-vs-equal-weight-out-of-sample-python.png"


def retry(fn, *a, **k):
    for attempt in range(5):
        try:
            return fn(*a, **k)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def year_panel(y):
    """Monthly total returns for 150 members of the S&P 500 as it stood on 1 January of year y."""
    path = f"{CACHE}/sp500_monthly_{y}.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    roster = retry(xfl.index, "sp500", as_of=f"{y}-01-01")
    ids = sorted({int(x) for x in roster["entity_id"].dropna()})
    pick = sorted(np.random.default_rng(y).choice(ids, size=150, replace=False).tolist())
    parts = [retry(xfl.prices, entity_id=pick[i:i + 50], start=f"{y - 5}-01-01",
                   end=f"{y}-12-31", interval="1mo", fields=["return_daily"],
                   max_rows=200000) for i in range(0, len(pick), 50)]
    px = pd.concat(parts, ignore_index=True)[["entity_id", "ticker", "date", "return_daily"]]
    px["year"] = y
    px.to_pickle(path)
    return px


def min_variance(S):
    """Long-only weights that minimise w'Sw subject to the weights summing to one."""
    n = len(S)
    r = minimize(lambda w: w @ S @ w, np.ones(n) / n, jac=lambda w: 2 * S @ w,
                 bounds=[(0.0, 1.0)] * n, method="SLSQP",
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                               "jac": lambda w: np.ones(n)}],
                 options={"maxiter": 400, "ftol": 1e-12})
    w = np.clip(r.x, 0, None)
    return w / w.sum(), r.success


def max_sharpe(mu, S):
    """Long-only tangency weights, solved in Kuhn-Tucker form: min y'Sy s.t. mu'y = 1, y >= 0."""
    n = len(mu)
    y0 = np.maximum(mu, 0.0)
    y0 = y0 / (y0 @ mu) if y0 @ mu > 0 else np.ones(n) / n
    r = minimize(lambda y: y @ S @ y, y0, jac=lambda y: 2 * S @ y,
                 bounds=[(0.0, None)] * n, method="SLSQP",
                 constraints=[{"type": "eq", "fun": lambda y: y @ mu - 1.0,
                               "jac": lambda y: mu}],
                 options={"maxiter": 400, "ftol": 1e-12})
    y = np.clip(r.x, 0, None)
    if y.sum() <= 0:
        return np.ones(n) / n, False
    return y / y.sum(), r.success


rows, roster_log = [], []
for y in YEARS:
    d = year_panel(y)
    d["m"] = pd.to_datetime(d["date"]).dt.to_period("M")
    R = (d.drop_duplicates(["entity_id", "m"])
          .pivot(index="m", columns="entity_id", values="return_daily").sort_index())
    est, oos = R[R.index.year <= y - 1], R[R.index.year == y]
    assert len(est) == 60 and len(oos) == 12, (y, len(est), len(oos))
    pool = [c for c in R.columns if est[c].notna().all()]
    E, O = est[pool].values, oos[pool].values
    roster_log.append((y, len(pool), int(np.isnan(O).sum())))
    for n in SIZES:
        for k in range(DRAWS):
            sel = np.random.default_rng(y * 1000 + n * 10 + k).choice(len(pool), n, replace=False)
            Xe, Xo = E[:, sel], O[:, sel]
            mu, S = Xe.mean(0), np.cov(Xe, rowvar=False)
            book = {"1/N": (np.ones(n) / n, True)}
            book["Minimum variance"] = min_variance(S)
            book["Maximum Sharpe"] = max_sharpe(mu, S)
            live = ~np.isnan(Xo)
            for name, (w, ok) in book.items():
                held = (live * w).sum(axis=1)
                r_p = np.where(held > 0,
                               np.nansum(np.where(live, Xo, 0.0) * w, axis=1) / np.where(held > 0, held, 1),
                               np.nan)
                rows.append({"year": y, "n": n, "draw": k, "book": name,
                             "in_sample": (w @ mu) / np.sqrt(w @ S @ w) * np.sqrt(12),
                             "eff_names": 1.0 / (w ** 2).sum(), "top_weight": w.max(),
                             "ok": ok, "rets": r_p})

df = pd.DataFrame(rows)
flat = df.explode("rets")
flat["rets"] = flat["rets"].astype(float)


def path_stats(r):
    r = np.asarray(r, dtype=float)
    return ((1 + r).prod() ** (12 / len(r)) - 1,
            r.std(ddof=1) * np.sqrt(12),
            r.mean() / r.std(ddof=1) * np.sqrt(12))


summary = []
for n in SIZES:
    for name in ["1/N", "Minimum variance", "Maximum Sharpe"]:
        g = df[(df["n"] == n) & (df["book"] == name)]
        paths = [path_stats(np.concatenate(gg["rets"].values))
                 for _, gg in g.sort_values("year").groupby("draw")]
        pooled = np.mean(np.stack([np.concatenate(gg.sort_values("year")["rets"].values)
                                   for _, gg in g.groupby("draw")]), axis=0)
        ar, av, sr = path_stats(pooled)
        summary.append({"n": n, "book": name, "ann_ret": ar, "ann_vol": av, "sharpe": sr,
                        "sr_lo": min(p[2] for p in paths), "sr_hi": max(p[2] for p in paths),
                        "in_sample": g["in_sample"].mean(), "eff_names": g["eff_names"].mean(),
                        "top_weight": g["top_weight"].mean()})
S = pd.DataFrame(summary)

W = 84
print("=" * W)
print("WHAT MEAN-VARIANCE OPTIMISATION PROMISES AND WHAT IT DELIVERS")
print("=" * W)
print("Universe   S&P 500 point-in-time roster on 1 January of each year, 2006-2024,")
print("           carried by entity id; 150 members drawn per year, of which those with a")
print("           complete 60-month return history form the pool")
print("Estimation 60 monthly total returns ending the December before each rebalance")
print("Holding    the next 12 months, weights held at target, 20 random draws per size")
print("Sharpe     annualised, zero risk-free rate, on the pooled 228-month series")
print()
print(f"Pool of complete-history names per year: min {min(r[1] for r in roster_log)}, "
      f"median {int(np.median([r[1] for r in roster_log]))}, max {max(r[1] for r in roster_log)}")
print(f"Optimiser convergence: {df['ok'].mean() * 100:.1f}%   "
      f"portfolio months with no priced holding: {int(flat['rets'].isna().sum())}")
print()
for n in SIZES:
    print(f"Portfolios of {n} stocks, 60 months of history")
    print(f"  {'':18s}{'return':>9s}{'vol':>9s}{'Sharpe':>9s}{'draw range':>16s}"
          f"{'promised SR':>13s}{'eff names':>11s}{'top wt':>9s}")
    for name in ["1/N", "Minimum variance", "Maximum Sharpe"]:
        r = S[(S["n"] == n) & (S["book"] == name)].iloc[0]
        print(f"  {name:18s}{r.ann_ret:8.2%}{r.ann_vol:9.2%}{r.sharpe:9.3f}"
              f"{r.sr_lo:8.2f} to{r.sr_hi:5.2f}{r.in_sample:13.2f}"
              f"{r.eff_names:11.1f}{r.top_weight:9.1%}")
    print()

# paired: each draw holds the same stocks under all three rules, so differences pair off
def draw_sharpes(n, name):
    g = df[(df["n"] == n) & (df["book"] == name)].sort_values("year")
    return {k: path_stats(np.concatenate(gg["rets"].values))[2] for k, gg in g.groupby("draw")}


print("Sharpe against the same stocks held in equal weight, draw by draw")
print(f"  {'stocks':>8s}{'minimum variance':>26s}{'maximum Sharpe':>26s}{'means add':>14s}")
for n in SIZES:
    e, m, x = draw_sharpes(n, "1/N"), draw_sharpes(n, "Minimum variance"), draw_sharpes(n, "Maximum Sharpe")
    dm = np.array([m[k] - e[k] for k in e])
    dx = np.array([x[k] - e[k] for k in e])
    du = np.array([x[k] - m[k] for k in e])
    print(f"  {n:8d}{dm.mean():+10.3f} ({(dm > 0).sum():2d}/{len(dm)} draws)"
          f"{dx.mean():+13.3f} ({(dx > 0).sum():2d}/{len(dx)} draws){du.mean():+14.3f}")
print("  ('means add' is maximum Sharpe minus minimum variance: the payoff to estimating")
print("   expected returns rather than using the covariance matrix alone)")
print("=" * W)

# ---------------------------------------------------------------- chart
BG, FG, ACC = "#0a0a0a", "#e0e0e0", "#3b82f6"
style = {"1/N": ("#9ca3af", "o"), "Minimum variance": ("#f59e0b", "s"),
         "Maximum Sharpe": (ACC, "D")}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, facecolor=BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.grid(axis="y", color="#1f1f1f", linewidth=0.8)
    ax.set_axisbelow(True)
pos = np.arange(len(SIZES))
for name, (c, mk) in style.items():
    g = S[S["book"] == name].sort_values("n")
    ax1.plot(pos, g["in_sample"], color=c, marker=mk, linewidth=2, label=name)
    ax2.plot(pos, g["sharpe"], color=c, marker=mk, linewidth=2, label=name)
ax1.set_ylabel("Sharpe ratio inside the\nestimation window", color=FG)
ax1.set_ylim(0.6, 2.9)
ax2.set_ylabel("Sharpe ratio over the\nnext twelve months", color=FG)
ax2.set_ylim(0.58, 0.85)
ax2.set_xlabel("Number of stocks held, each estimated from 60 months of returns", color=FG)
ax2.set_xticks(pos)
ax2.set_xticklabels([str(s) for s in SIZES])
ax1.legend(facecolor=BG, edgecolor="#333333", labelcolor=FG, framealpha=1, loc="upper left")
fig.suptitle("What the optimiser promised, and what it delivered", color=FG, fontsize=14)
plt.tight_layout()
plt.savefig(IMG, dpi=150, facecolor=BG)
print("chart:", IMG)

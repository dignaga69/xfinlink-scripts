# Full write-up: https://xfinlink.com/blog/beta-cap-weighted-vs-equal-weighted-market-python
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xfinlink as xfl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2023-09-05", "2026-09-04"


def retry(fn, *a, **k):
    for attempt in range(5):
        try:
            return fn(*a, **k)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(4 * (attempt + 1))


# companies are carried by entity id, so a retired or reused ticker cannot swap one for another
roster = sorted(xfl.index("sp500", as_of=START)["entity_id"].dropna().astype(int).unique())


def grab(i):
    try:
        return retry(xfl.prices, entity_id=i, start=START, end=END, interval="1w",
                     fields=["return_daily"])
    except Exception:
        return pd.DataFrame(columns=["entity_id", "ticker", "date", "return_daily"])


with ThreadPoolExecutor(4) as ex:
    px = pd.concat(ex.map(grab, roster), ignore_index=True)

R = px.drop_duplicates(["entity_id", "date"]).pivot(index="date", columns="entity_id",
                                                    values="return_daily")
R = R[R.notna().sum(axis=1) >= 0.9 * R.shape[1]]          # weeks the roster trades in
R = R[R.columns[R.notna().sum() == len(R)]]               # companies present in all of them

# cap weights are reset to filed share counts on the first trading week of each quarter
resets = list(pd.Series(R.index, index=R.index).groupby([R.index.year, R.index.quarter]).min())
cols = list(R.columns)


def caps_on(d):
    lo = (d - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    hi = (d + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    last = lambda m: (m.dropna(subset=["market_cap"]).sort_values("period_end")
                      .groupby("entity_id")["market_cap"].last().to_dict())
    out = {}
    for i in range(0, len(cols), 100):
        out.update(last(retry(xfl.metrics, entity_id=cols[i:i + 100], period_type="daily",
                              fields=["market_cap"], start=lo, end=hi, max_rows=100000)))
    for e in [c for c in cols if c not in out]:           # repair anything a batch missed
        out.update(last(retry(xfl.metrics, entity_id=int(e), period_type="daily",
                              fields=["market_cap"], start=lo, end=hi)))
    return pd.Series(out)


CAP = pd.DataFrame({d: caps_on(d) for d in resets}).T
CAP = CAP[[c for c in cols if c in CAP.columns]].dropna(axis=1)
R = R[CAP.columns]
name = px.dropna(subset=["ticker"]).groupby("entity_id")["ticker"].last()

# inside a quarter the cap portfolio is buy-and-hold, so weights drift with price
V = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
VP = V.copy()
for i, d in enumerate(resets):
    upper = resets[i + 1] if i + 1 < len(resets) else R.index[-1]
    seg = R.index[(R.index > d) & (R.index <= upper)]
    c, g = CAP.loc[d].values, np.cumprod(1.0 + R.loc[seg].values, axis=0)
    V.loc[seg], VP.loc[seg] = c * g, c * np.vstack([np.ones(len(c)), g[:-1]])

keep = V.notna().all(axis=1)
Vv, VPv, Rv = V[keep].values, VP[keep].values, R[keep].values
n, dates = Rv.shape[1], R.index[keep]

# every company is measured against a market that excludes it
cap_ex = (Vv.sum(1, keepdims=True) - Vv) / (VPv.sum(1, keepdims=True) - VPv) - 1.0
ew_ex = (Rv.sum(1, keepdims=True) - Rv) / (n - 1)
cap_mkt, ew_mkt = Vv.sum(1) / VPv.sum(1) - 1.0, Rv.mean(1)


def fit(y, x):
    xm, ym = x - x.mean(0), y - y.mean(0)
    b = (xm * ym).sum(0) / (xm * xm).sum(0)
    r2 = b ** 2 * (xm * xm).sum(0) / (ym * ym).sum(0)
    return b, r2, (ym - b * xm).std(0, ddof=2) * np.sqrt(52)


b_cap, r2_cap, rv_cap = fit(Rv, cap_ex)
b_ew, r2_ew, rv_ew = fit(Rv, ew_ex)

res = pd.DataFrame({"cap_bn": CAP.loc[resets[0]].values / 1000.0,
                    "b_cap": b_cap, "b_ew": b_ew, "r2_cap": r2_cap, "r2_ew": r2_ew,
                    "rv_cap": rv_cap, "rv_ew": rv_ew,
                    "vol": Rv.std(0, ddof=1) * np.sqrt(52)}, index=name[R.columns].values)
res["wedge"] = res["b_ew"] - res["b_cap"]
res["dec"] = pd.qcut(res["cap_bn"].rank(method="first"), 10, labels=False) + 1
dec = res.groupby("dec").agg(n=("b_cap", "size"), cap=("cap_bn", "median"),
                             b_cap=("b_cap", "mean"), b_ew=("b_ew", "mean"),
                             r2_cap=("r2_cap", "mean"), r2_ew=("r2_ew", "mean"),
                             rv_cap=("rv_cap", "mean"), rv_ew=("rv_ew", "mean"))

h = len(Rv) // 2
w1 = fit(Rv[:h], ew_ex[:h])[0] - fit(Rv[:h], cap_ex[:h])[0]
w2 = fit(Rv[h:], ew_ex[h:])[0] - fit(Rv[h:], cap_ex[h:])[0]
stab = pd.Series(w1).corr(pd.Series(w2), method="spearman")

# ----------------------------------------------------------------- report
W = 84
print("=" * W)
print("ONE COMPANY, TWO BETAS: CAP-WEIGHTED AND EQUAL-WEIGHTED MARKETS")
print("=" * W)
print(f"Roster     S&P 500 as at {START}: {len(roster)} companies, of which {len(res)} carry a")
print(f"           complete weekly return series through {END}")
print(f"Sample     {len(dates)} weekly returns, {dates.min().date()} to {dates.max().date()}")
print(f"Weights    cap weights reset to filed share counts on {len(resets)} quarter-opening weeks,")
print("           drifting with price in between")
print("Betas      ordinary least squares on weekly price returns; each company is dropped")
print("           from the market it is measured against")
print()
print("The two markets, built from the same companies")
print(f"  {'':30}{'cap-weighted':>14}{'equal-weighted':>16}")
print(f"  {'Total return':30}{(np.prod(1+cap_mkt)-1)*100:13.2f}%{(np.prod(1+ew_mkt)-1)*100:15.2f}%")
print(f"  {'Annualised volatility':30}{cap_mkt.std(ddof=1)*np.sqrt(52)*100:13.2f}%"
      f"{ew_mkt.std(ddof=1)*np.sqrt(52)*100:15.2f}%")
print(f"  {'Largest weight at the end':30}"
      f"{V[keep].iloc[-1].max()/V[keep].iloc[-1].sum()*100:13.2f}%{100/n:15.2f}%")
print(f"  Correlation of the two weekly return series           "
      f"{np.corrcoef(cap_mkt, ew_mkt)[0,1]:.4f}")
print()
print("Cross-section of the two betas")
print(f"  {'':22}{'mean':>8}{'median':>9}{'p10':>8}{'p90':>8}")
for lab, c in (("Beta, cap-weighted", "b_cap"), ("Beta, equal-weighted", "b_ew"),
               ("Equal minus cap", "wedge")):
    q = res[c]
    print(f"  {lab:22}{q.mean():8.3f}{q.median():9.3f}{q.quantile(.1):8.3f}{q.quantile(.9):8.3f}")
print(f"  Companies whose equal-weighted beta is the higher of the two: "
      f"{(res['wedge']>0).mean()*100:.1f}%")
print(f"  Rank correlation between the two betas: "
      f"{res['b_cap'].corr(res['b_ew'], method='spearman'):.3f}")
print()
print("Share of a company's return variance the market explains")
print(f"  {'':22}{'mean':>8}{'median':>9}")
for lab, c in (("Cap-weighted", "r2_cap"), ("Equal-weighted", "r2_ew")):
    print(f"  {lab:22}{res[c].mean():8.3f}{res[c].median():9.3f}")
print(f"  Companies better explained by the equal-weighted market: "
      f"{(res['r2_ew']>res['r2_cap']).mean()*100:.1f}%")
print(f"  Average annualised risk left after a beta hedge: "
      f"{res['rv_cap'].mean()*100:.1f}% against the cap-weighted market, "
      f"{res['rv_ew'].mean()*100:.1f}% against the equal-weighted one")
print()
print("By market-capitalisation decile at the start of the window")
print(f"  {'Dec':>4}{'n':>4}{'median cap':>13}{'beta cap':>10}{'beta EW':>9}"
      f"{'R2 cap':>9}{'R2 EW':>8}{'left cap':>10}{'left EW':>9}")
for i, r in dec.iterrows():
    print(f"  {i:>4}{int(r['n']):>4}{r['cap']:>11.0f}bn{r['b_cap']:>10.3f}{r['b_ew']:>9.3f}"
          f"{r['r2_cap']:>9.3f}{r['r2_ew']:>8.3f}{r['rv_cap']*100:>9.1f}%{r['rv_ew']*100:>8.1f}%")
print("  (1 = smallest, 10 = largest; 'left' is annualised residual volatility after the hedge)")
print()
print(f"  {'Largest 8':12}{'cap':>10}{'beta cap':>10}{'beta EW':>9}{'R2 cap':>9}{'R2 EW':>8}")
for t, r in res.sort_values("cap_bn", ascending=False).head(8).iterrows():
    print(f"  {t:12}{r['cap_bn']:>8.0f}bn{r['b_cap']:>10.3f}{r['b_ew']:>9.3f}"
          f"{r['r2_cap']:>9.3f}{r['r2_ew']:>8.3f}")
print(f"  {'Smallest 8':12}")
for t, r in res.sort_values("cap_bn").head(8).iterrows():
    print(f"  {t:12}{r['cap_bn']:>8.0f}bn{r['b_cap']:>10.3f}{r['b_ew']:>9.3f}"
          f"{r['r2_cap']:>9.3f}{r['r2_ew']:>8.3f}")
print()
print("Does the gap between the two betas persist?")
print(f"  Rank correlation of (equal minus cap) across the two halves of the window: {stab:.3f}")
print(f"  Mean gap, first half {w1.mean():.3f}   second half {w2.mean():.3f}")
print("=" * W)

# ------------------------------------------------------------------ chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 10})
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
x = np.arange(len(dec))
for a, (ca, ce, ylab) in zip(ax, [("b_cap", "b_ew", "Average beta"),
                                  ("r2_cap", "r2_ew", "Share of variance explained")]):
    a.bar(x - 0.19, dec[ca], 0.38, color="#3b82f6", label="Cap-weighted market")
    a.bar(x + 0.19, dec[ce], 0.38, color="#f59e0b", label="Equal-weighted market")
    a.set_ylabel(ylab)
    a.spines[["top", "right"]].set_visible(False)
ax[0].axhline(1.0, color="#666666", lw=0.8, ls="--")
ax[0].set_title("One company, two betas: S&P 500 members against two versions of their own market",
                color="#e0e0e0", fontsize=12, pad=12)
ax[0].legend(frameon=False, loc="upper right")
ax[1].set_xticks(x)
ax[1].set_xticklabels([str(i) for i in dec.index])
ax[1].set_xlabel("Market-capitalisation decile at the start of the window "
                 "(1 = smallest, 10 = largest)")
plt.tight_layout()
plt.savefig("beta-cap-weighted-vs-equal-weighted-market-python.png", dpi=150,
            facecolor="#0a0a0a")

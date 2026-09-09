# Full write-up: https://xfinlink.com/blog/does-mean-reversion-survive-trading-costs-python
#
# Deviations from a 50-day moving average: stationarity, half-life, and whether
# the resulting signal pays for itself once the market's own move is removed.

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

START, END = "2020-01-01", "2024-12-31"
MA, SD, HOLD, ZLIM = 50, 250, 5, -1.5          # average, z window, hold, trigger
COST = 0.0020                                   # 20 bp round trip

# ---- universe: S&P 500 as it stood at 2019-12-31, carried by entity id -------
roster = xfl.index("sp500", as_of="2019-12-31")
ids = sorted({int(e) for e in roster["entity_id"].dropna()})
px = pd.concat(
    [xfl.prices(entity_id=ids[i:i + 20], start=START, end=END,
                fields=["adj_close"], max_rows=200000)
     for i in range(0, len(ids), 20)],
    ignore_index=True)
px = px[px["adj_close"] > 0].dropna(subset=["adj_close"])

# ---- per company: stationarity, half-life, forward returns ------------------
rows, fwds, sigs = [], {}, {}
for eid, g in px.groupby("entity_id"):
    g = g.sort_values("date").drop_duplicates("date")
    if len(g) < 1200:                                    # complete series only
        continue
    p = np.log(g["adj_close"].to_numpy())
    if np.max(np.abs(np.diff(p))) > 0.40:                # corporate-action moves
        continue
    s = pd.Series(p, index=g["date"].to_numpy())
    dev = s - s.rolling(MA).mean()                       # log price less its average
    z = dev / dev.rolling(SD).std()
    fwd = np.expm1(s.shift(-HOLD) - s)                   # forward 5-session return
    valid = dev.notna() & z.notna() & fwd.notna()
    if int(valid.sum()) < 500:
        continue
    d = dev[valid].to_numpy()
    b = np.polyfit(d[:-1], d[1:], 1)[0]                  # AR(1) on the deviation
    sig = valid & (z <= ZLIM)
    if int(sig.sum()) < 20:
        continue
    fwds[eid], sigs[eid] = fwd.where(valid), sig
    rows.append(dict(
        entity_id=eid, ticker=g["ticker"].iloc[-1],
        adf_p_dev=adfuller(d, autolag="AIC")[1],
        adf_p_px=adfuller(s.to_numpy(), autolag="AIC")[1],
        ar1=b, half_life=-np.log(2) / np.log(b) if 0 < b < 1 else np.nan,
        n_sig=int(sig.sum()), mu_sig=float(fwd[sig].mean()),
        mu_all=float(fwd[valid].mean())))

res = pd.DataFrame(rows).set_index("entity_id")

# ---- remove the market's own move on the same days --------------------------
F = pd.DataFrame(fwds)                                   # date x company
S = pd.DataFrame(sigs).reindex_like(F).fillna(False).astype(bool)
X = F.sub(F.mean(axis=1), axis=0)                        # less the equal-weight cross-section
res["edge"] = res.mu_sig - res.mu_all
res["mu_sig_x"] = [X[e][S[e]].mean() for e in res.index]
res["edge_x"] = res.mu_sig_x - [X[e][F[e].notna()].mean() for e in res.index]
res = res.reset_index()

t = lambda v: v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
res["hlq"] = pd.qcut(res.half_life, 4, labels=["Q1 fastest", "Q2", "Q3", "Q4 slowest"])
byq = res.groupby("hlq", observed=True).agg(
    n=("edge", "size"), hl=("half_life", "median"),
    raw=("edge", "mean"), neu=("edge_x", "mean"))
ev = pd.DataFrame({"raw": F.where(S).stack(), "neu": X.where(S).stack()}).dropna()
ev.index.names = ["date", "entity_id"]
byy = ev.reset_index().assign(year=lambda d: d["date"].dt.year).groupby("year").agg(
    events=("raw", "size"), raw=("raw", "mean"), neu=("neu", "mean"))

# ---- output -----------------------------------------------------------------
sd = S.stack()
sd = sd[sd].index.get_level_values(0)
print(f"S&P 500 at 2019-12-31 carried by entity id | priced {START} to {END}")
print(f"{len(res)} companies after screens | {int(res.n_sig.sum()):,} signals, "
      f"{sd.min().date()} to {sd.max().date()} | {int(F.notna().sum().sum()):,} company-days")
print()
print("Augmented Dickey-Fuller, stationary at 5%")
print(f"  log price                        {(res.adf_p_px < 0.05).sum():3d} of {len(res)}")
print(f"  deviation from 50-day average    {(res.adf_p_dev < 0.05).sum():3d} of {len(res)}")
print()
print("Half-life of a deviation, AR(1), trading sessions")
print(f"  median {res.half_life.median():.1f}   quartiles {res.half_life.quantile(.25):.1f} "
      f"to {res.half_life.quantile(.75):.1f}   range {res.half_life.min():.1f} to {res.half_life.max():.1f}")
print()
print("Forward 5-session price return after the deviation reaches -1.5 sd")
print(f"  every day, average company            {res.mu_all.mean()*1e4:7.1f} bp")
print(f"  after a signal                        {res.mu_sig.mean()*1e4:7.1f} bp")
print(f"  edge                                  {res.edge.mean()*1e4:7.1f} bp   "
      f"t {t(res.edge):5.2f}   {(res.edge > 0).mean()*100:.1f}% of companies positive")
print(f"  edge less the market's move           {res.edge_x.mean()*1e4:7.1f} bp   "
      f"t {t(res.edge_x):5.2f}   {(res.edge_x > 0).mean()*100:.1f}% of companies positive")
print(f"  same, net of a {COST*1e4:.0f} bp round trip      {(res.edge_x - COST).mean()*1e4:7.1f} bp   "
      f"{'':10s}{(res.edge_x > COST).mean()*100:.1f}% of companies positive")
print()
print("By half-life quartile                    raw edge   less market")
for q, r in byq.iterrows():
    print(f"  {q:<11s} {r.n:3.0f} names, median {r.hl:4.1f}   {r.raw*1e4:7.1f} bp   {r.neu*1e4:7.1f} bp")
print()
print("By signal year                           raw edge   less market")
for y, r in byy.iterrows():
    print(f"  {y}  {r.events:6,.0f} signals            {r.raw*1e4:7.1f} bp   {r.neu*1e4:7.1f} bp")

# ---- chart ------------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.hist(res.half_life, bins=32, color="#3b82f6", edgecolor="#0a0a0a")
ax1.axvline(res.half_life.median(), color="#e0e0e0", ls="--", lw=1)
ax1.text(res.half_life.median() + 0.6, ax1.get_ylim()[1] * 0.88,
         f"median {res.half_life.median():.1f} sessions", color="#e0e0e0")
ax1.set_xlabel("Half-life of a deviation from the 50-day average (trading sessions)")
ax1.set_ylabel("Companies")
ax1.set_title("How long a stretched price takes to close half the gap", loc="left")
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)

w, xs = 0.38, np.arange(len(byq))
ax2.bar(xs - w / 2, byq.raw * 1e4, w, color="#3b82f6", label="Before removing the market's move")
ax2.bar(xs + w / 2, byq.neu * 1e4, w, color="#94a3b8", label="After removing the market's move")
ax2.axhline(COST * 1e4, color="#e0e0e0", ls="--", lw=1)
ax2.text(len(byq) - 0.45, COST * 1e4 + 3, "20 bp round-trip cost", color="#e0e0e0", ha="right")
ax2.axhline(0, color="#3f3f3f", lw=1)
ax2.set_xticks(xs)
ax2.set_xticklabels([f"{q}\nmedian {h:.1f} sessions" for q, h in zip(byq.index, byq.hl)])
ax2.set_ylabel("Five-session edge (basis points)")
ax2.set_title("Payoff after a 1.5 standard deviation stretch, by half-life quartile", loc="left")
ax2.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0", loc="upper right")
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig("does-mean-reversion-survive-trading-costs-python.png", dpi=150,
            facecolor="#0a0a0a")

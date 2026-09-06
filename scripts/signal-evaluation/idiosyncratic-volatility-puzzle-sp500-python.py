# Full write-up: https://xfinlink.com/blog/idiosyncratic-volatility-puzzle-sp500-python
"""Do high-idiosyncratic-volatility S&P 500 stocks underperform?

Sorts the point-in-time S&P 500 into quintiles on residual volatility from a
market model, measures the next month's equal-weighted return, and checks
whether the sort is anything other than a total-volatility sort in disguise.
"""

from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

WIN, MIN_OBS, NQ = 252, 240, 5
SLUG = "idiosyncratic-volatility-puzzle-sp500-python"

# ---------------------------------------------------------------- data --------
months = pd.date_range("2015-01-31", "2025-12-31", freq="ME")
roster = {d: xfl.index("sp500", as_of=d.strftime("%Y-%m-%d"))["entity_id"].tolist()
          for d in months}
ids = sorted({e for v in roster.values() for e in v})
chunks = [ids[i:i + 90] for i in range(0, len(ids), 90)]

grab = lambda cy: xfl.prices(entity_id=cy[0], start=f"{cy[1]}-01-01", end=f"{cy[1]}-12-31",
                             fields=["adj_close"], max_rows=60000)
with ThreadPoolExecutor(4) as ex:
    px = pd.concat(ex.map(grab, [(c, y) for y in range(2015, 2026) for c in chunks]))
spy = xfl.prices("SPY", start="2015-01-01", end="2025-12-31", fields=["adj_close"])

mkt = spy.set_index("date")["adj_close"].sort_index()
P = px.drop_duplicates(["entity_id", "date"]).pivot(index="date", columns="entity_id",
                                                    values="adj_close")
R = P.sort_index().reindex(mkt.index).pct_change()
n_all = R.shape[1]
# a daily move above +100% or below -50% is a corporate action, not a return
R = R.drop(columns=R.columns[((R > 1.0) | (R < -0.5)).any()])
n_drop = n_all - R.shape[1]
M, col = mkt.pct_change().values, {c: i for i, c in enumerate(R.columns)}

sd = [(d, R.index.searchsorted(d, "right") - 1) for d in months]
sd = [(d, p) for d, p in sd if p >= WIN and R.index[p].to_period("M") == d.to_period("M")]

rank = lambda x, q: np.argsort(np.argsort(x)) * q // len(x)
tstat = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


def build(pairs):
    """For each (sort date, next date) pair return vol measures and forward returns."""
    out = []
    for (d, p), (_, nxt) in pairs:
        j = [col[e] for e in roster[d] if e in col]
        W, F, m = R.values[p - WIN + 1:p + 1, j], R.values[p + 1:nxt + 1, j], M[p - WIN + 1:p + 1]
        ok = ((~np.isnan(W)).sum(0) >= MIN_OBS) & ((~np.isnan(F)).sum(0) == F.shape[0])
        if ok.sum() < 100:
            continue
        W, F = np.nan_to_num(W[:, ok]), F[:, ok]
        md = m - m.mean()
        beta = (W - W.mean(0)).T @ md / (md @ md)
        resid = W - (W.mean(0) - beta * m.mean()) - np.outer(m, beta)
        out.append(dict(date=R.index[nxt], n=F.shape[1], beta=beta,
                        ivol=resid.std(0, ddof=2) * np.sqrt(252),
                        totvol=W.std(0, ddof=1) * np.sqrt(252),
                        fwd=np.prod(1 + F, axis=0) - 1))
    return out


rows = build(list(zip(sd[:-1], sd[1:])))
qtr = build(list(zip(sd[:-3:3], sd[3::3])))

# ------------------------------------------------------------ statistics ------
quintiles = {k: np.array([[r["fwd"][rank(r[k], NQ) == q].mean() for q in range(NQ)]
                          for r in rows]) for k in ("ivol", "totvol")}
spread = {k: v[:, -1] - v[:, 0] for k, v in quintiles.items()}
mean_vol = {k: np.array([[r[k][rank(r[k], NQ) == q].mean() for q in range(NQ)]
                         for r in rows]).mean(0) for k in ("ivol", "totvol")}

cond_iv = np.array([[r["fwd"][g][rank(r["ivol"][g], 3) == 2].mean()
                     - r["fwd"][g][rank(r["ivol"][g], 3) == 0].mean()
                     for g in [rank(r["totvol"], NQ) == q for q in range(NQ)]] for r in rows])
cond_tv = np.array([[r["fwd"][g][rank(r["totvol"][g], 3) == 2].mean()
                     - r["fwd"][g][rank(r["totvol"][g], 3) == 0].mean()
                     for g in [rank(r["ivol"], NQ) == q for q in range(NQ)]] for r in rows])

rho = np.array([pd.Series(r["ivol"]).corr(pd.Series(r["totvol"]), method="spearman") for r in rows])
rho_beta = np.array([pd.Series(r["ivol"]).corr(pd.Series(r["beta"]), method="spearman") for r in rows])
overlap = np.array([np.mean(np.isin(np.where(rank(r["ivol"], NQ) == NQ - 1)[0],
                                    np.where(rank(r["totvol"], NQ) == NQ - 1)[0])) for r in rows])

# ---------------------------------------------------------------- report ------
L = []
L.append("Idiosyncratic volatility and next-month returns, point-in-time S&P 500")
L.append(f"Panel:    {len(rows)} monthly sorts, {rows[0]['date']:%Y-%m} to {rows[-1]['date']:%Y-%m}; "
         f"{n_all} companies appear on a roster,")
L.append(f"          {n_drop} drop on the daily-return screen; "
         f"median {int(np.median([r['n'] for r in rows]))} names ranked per sort "
         f"(range {min(r['n'] for r in rows)} to {max(r['n'] for r in rows)})")
L.append("Formation: 252 trading days of daily price returns, market model against SPY")
L.append("Holding:  equal-weighted, one month, no skip")
L.append("")
L.append("Average next-month return by quintile (1 = lowest volatility, 5 = highest)")
L.append("  Quintile              1       2       3       4       5    5 minus 1        t")
for k, label in (("ivol", "Idiosyncratic vol"), ("totvol", "Total vol")):
    L.append(f"  {label:<18}" + "".join(f"{v*100:7.2f}%" for v in quintiles[k].mean(0)) +
             f"{spread[k].mean()*100:11.2f}%{tstat(spread[k]):9.2f}")
L.append("")
L.append("  Average annualised volatility in each quintile")
L.append("  Idiosyncratic vol " + "".join(f"{v*100:7.1f}%" for v in mean_vol["ivol"]))
L.append("  Total vol         " + "".join(f"{v*100:7.1f}%" for v in mean_vol["totvol"]))
L.append("")
L.append("Annualised long-short return (quintile 5 minus quintile 1)")
for k, label in (("ivol", "Idiosyncratic vol"), ("totvol", "Total vol")):
    s = spread[k]
    L.append(f"  {label:<18}{((1+s.mean())**12-1)*100:8.2f}%   monthly {s.mean()*100:6.2f}%   "
             f"t = {tstat(s):5.2f}   hit rate {(s > 0).mean()*100:4.1f}%")
L.append("")
L.append("Are the two sorts the same sort?")
L.append(f"  Cross-sectional rank correlation, idiosyncratic vs total vol   {rho.mean():.3f} "
         f"({rho.min():.3f} to {rho.max():.3f})")
L.append(f"  Top quintile shared by both rankings                           {overlap.mean()*100:.1f}%")
L.append(f"  Rank correlation of idiosyncratic vol with market beta         {rho_beta.mean():.3f}")
L.append("  Correlation of the two monthly long-short return series        "
         f"{np.corrcoef(spread['ivol'], spread['totvol'])[0, 1]:.3f}")
L.append("")
L.append("How wide is the uncertainty on the monthly long-short return?")
for k, label in (("ivol", "Idiosyncratic vol"), ("totvol", "Total vol")):
    s = spread[k]
    se = s.std(ddof=1) / np.sqrt(len(s))
    L.append(f"  {label:<18}{s.mean()*100:6.2f}%   standard error {se*100:.2f}%   "
             f"95% interval {(s.mean()-1.98*se)*100:6.2f}% to {(s.mean()+1.98*se)*100:5.2f}%")
L.append("")
L.append("Holding for a quarter instead of a month, non-overlapping")
for k, label in (("ivol", "Idiosyncratic vol"), ("totvol", "Total vol")):
    q = np.array([[r["fwd"][rank(r[k], NQ) == i].mean() for i in range(NQ)] for r in qtr])
    s = q[:, -1] - q[:, 0]
    L.append(f"  {label:<18}{len(qtr)} quarters   quintile 5 minus 1 {s.mean()*100:6.2f}%   "
             f"t = {tstat(s):5.2f}")
L.append("")
L.append("Double sort: hold one dimension fixed, sort on the other")
L.append("  Idiosyncratic vol high minus low, within total-vol quintiles   "
         f"{cond_iv.mean()*100:6.2f}% a month   t = {tstat(cond_iv.mean(1)):5.2f}")
L.append("  Total vol high minus low, within idiosyncratic-vol quintiles   "
         f"{cond_tv.mean()*100:6.2f}% a month   t = {tstat(cond_tv.mean(1)):5.2f}")
L.append("")
L.append("  Idiosyncratic-vol spread inside each total-vol quintile")
L.append("  Total-vol quintile    1       2       3       4       5")
L.append("  High minus low  " + "".join(f"{v*100:7.2f}%" for v in cond_iv.mean(0)))
L.append("")
L.append("Long-short return by calendar year (quintile 5 minus quintile 1)")
yr = pd.DataFrame({"date": [r["date"] for r in rows], "ivol": spread["ivol"],
                   "totvol": spread["totvol"]})
L.append("  Year    Idio vol   Total vol")
for y, g in yr.groupby(yr["date"].dt.year):
    L.append(f"  {y}    {((1+g['ivol']).prod()-1)*100:7.2f}%    {((1+g['totvol']).prod()-1)*100:7.2f}%")
print("\n".join(L))

# ----------------------------------------------------------------- chart ------
BG, FG, AC, AC2 = "#0a0a0a", "#e0e0e0", "#3b82f6", "#f59e0b"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=BG,
                               gridspec_kw={"height_ratios": [1, 1.1]})
x, w = np.arange(NQ), 0.38
ax1.bar(x - w / 2, quintiles["ivol"].mean(0) * 100, w, color=AC,
        label="Idiosyncratic volatility sort")
ax1.bar(x + w / 2, quintiles["totvol"].mean(0) * 100, w, color=AC2,
        label="Total volatility sort")
ax1.set_xticks(x)
ax1.set_xticklabels(["1 (lowest)", "2", "3", "4", "5 (highest)"])
ax1.set_ylabel("Average next-month return (%)")
ax1.set_title("Next-month return by volatility quintile, S&P 500, 2016-2025", color=FG, fontsize=12)
ax1.legend(frameon=False, labelcolor=FG, fontsize=9)

dates = pd.to_datetime([r["date"] for r in rows])
ax2.plot(dates, np.cumprod(1 + spread["ivol"]), color=AC, lw=1.6,
         label="High minus low idiosyncratic volatility")
ax2.plot(dates, np.cumprod(1 + spread["totvol"]), color=AC2, lw=1.6,
         label="High minus low total volatility")
ax2.axhline(1.0, color="#555555", lw=0.8)
ax2.set_ylabel("Value of $1 in the long-short spread")
ax2.set_title("Buying the highest-volatility quintile and selling the lowest", color=FG, fontsize=12)
ax2.legend(frameon=False, labelcolor=FG, fontsize=9)

for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#333333")
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)

plt.tight_layout(h_pad=2.5)
plt.savefig(f"{SLUG}.png", dpi=150, facecolor=BG)

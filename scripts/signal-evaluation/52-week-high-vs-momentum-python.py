# Full write-up: https://xfinlink.com/blog/52-week-high-vs-momentum-python
#
# Proximity to the 52-week high and 12-1 momentum both rank stocks on past
# prices, and the two rankings overlap. This sorts a point-in-time S&P 500
# panel on each score separately, then conditionally: quintiles of one score
# formed inside quintiles of the other, so each signal is measured with the
# other one held fixed. Returns are price returns computed from adj_close.

import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ANCHORS = pd.date_range("2016-01-31", "2026-07-31", freq="ME").strftime("%Y-%m-%d").tolist()
START, END = "2014-12-01", "2026-08-31"
FORM, SKIP, FLOOR, JUMP = 252, 21, 5.0, 0.40   # formation days, skip days, price floor, jump screen
PNG = "52-week-high-vs-momentum-python.png"

# ---------------------------------------------------------------- data ------


def roster(as_of):
    return sorted(int(e) for e in xfl.index("sp500", as_of=as_of)["entity_id"].dropna())


def series(entity):
    return xfl.prices(entity_id=entity, start=START, end=END,
                      fields=["close", "adj_close"], max_rows=200000)


with ThreadPoolExecutor(8) as pool:
    rosters = dict(zip(ANCHORS, pool.map(roster, ANCHORS)))
universe = sorted({e for ids in rosters.values() for e in ids})

with ThreadPoolExecutor(8) as pool:
    px = pd.concat([d for d in pool.map(series, universe) if len(d)], ignore_index=True)

px = px.drop_duplicates(["entity_id", "date"])
px = px[px["adj_close"] > 0]
P = px.pivot(index="date", columns="entity_id", values="adj_close").sort_index()

cal, V = P.index, P.values
ret = np.vstack([np.full((1, V.shape[1]), np.nan), V[1:] / V[:-1] - 1.0])   # daily price return
live = (~np.isnan(V)).cumsum(0)                          # cumulative count of traded days
jump = (np.abs(ret) > JUMP).cumsum(0)                    # cumulative count of outsized sessions
runmax = pd.DataFrame(V).rolling(FORM, min_periods=FORM).max().values
pos = {e: i for i, e in enumerate(P.columns)}
month_end = pd.Series(np.arange(len(cal))).groupby(np.asarray(cal.year * 100 + cal.month)).max()
days = [int(month_end[int(a[:4]) * 100 + int(a[5:7])]) for a in ANCHORS]

# ------------------------------------------------------------ the sorts -----

rows, counts, grid_a, grid_b, spear, ovl = [], [], [], [], [], []
q_pth, q_mom = [], []
for anchor, t, nxt in zip(ANCHORS, days, days[1:]):
    f, s = t - FORM + 1, t - SKIP        # formation start, skip point
    ids = np.array([pos[e] for e in rosters[anchor] if e in pos])
    ok = ((live[t, ids] - live[f - 1, ids] == t - f + 1)      # complete formation window
          & (live[nxt, ids] - live[t, ids] == nxt - t)        # traded through the holding month
          & (V[t, ids] >= FLOOR)                              # price floor at the sort date
          & (jump[t, ids] - jump[f - 1, ids] == 0))           # no formation session beyond +/-40%
    ids = ids[ok]
    n = len(ids)

    pth = V[t, ids] / runmax[t, ids]      # proximity to the 52-week high, 1.0 = sitting on it
    mom = V[s, ids] / V[f, ids] - 1.0     # 12-1 price momentum
    fwd = V[nxt, ids] / V[t, ids] - 1.0   # the month held, price return
    lo, hi = np.percentile(fwd, [1, 99])
    fw = np.clip(fwd, lo, hi)             # cross-sectional winsorisation

    quint = lambda x, m: np.argsort(np.argsort(x, kind="stable"), kind="stable") * 5 // m
    qp, qm = quint(pth, n), quint(mom, n)

    def inside(mask, score):              # rank one score inside a bucket of the other
        idx = np.where(mask)[0]
        q = np.full(n, -1)
        q[idx] = quint(score[idx], len(idx))
        return q

    A = np.zeros((5, 5))                  # rows momentum quintile, cols 52-week-high quintile in it
    B = np.zeros((5, 5))                  # rows 52-week-high quintile, cols momentum quintile in it
    for i in range(5):
        ai, bi = inside(qm == i, pth), inside(qp == i, mom)
        for j in range(5):
            A[i, j], B[i, j] = fw[ai == j].mean(), fw[bi == j].mean()

    rows.append((cal[nxt],
                 fw[qp == 4].mean() - fw[qp == 0].mean(),     # 52-week high, raw
                 fw[qm == 4].mean() - fw[qm == 0].mean(),     # momentum, raw
                 (A[:, 4] - A[:, 0]).mean(),                  # 52-week high, momentum held fixed
                 (B[:, 4] - B[:, 0]).mean(),                  # momentum, 52-week high held fixed
                 fw.mean(), fwd.mean(),
                 fwd[qp == 4].mean() - fwd[qp == 0].mean()))  # 52-week high, unwinsorised
    q_pth.append([fw[qp == k].mean() for k in range(5)])
    q_mom.append([fw[qm == k].mean() for k in range(5)])
    grid_a.append(A)
    grid_b.append(B)
    counts.append(n)
    spear.append(stats.spearmanr(pth, mom).statistic)
    ovl.append(len(set(ids[qp == 4]) & set(ids[qm == 4])) / (qp == 4).sum())

out = pd.DataFrame(rows, columns=["date", "pth", "mom", "pth_n", "mom_n",
                                  "univ", "univ_raw", "pth_raw"]).set_index("date")
GA, GB = np.array(grid_a).mean(0), np.array(grid_b).mean(0)
QP, QM = np.array(q_pth).mean(0), np.array(q_mom).mean(0)
N = len(out)


def stat(x):
    x = np.asarray(x)
    return {"mean": x.mean(), "ann": np.prod(1 + x) ** (12 / len(x)) - 1,
            "vol": x.std(ddof=1) * np.sqrt(12),
            "t": x.mean() / (x.std(ddof=1) / np.sqrt(len(x))),
            "hit": (x > 0).mean(), "dollar": np.prod(1 + x)}


# --------------------------------------------------------------- print ------

print("Proximity to the 52-week high against 12-1 momentum, point-in-time S&P 500")
print(f"Panel:    {len(ANCHORS)} month-end rosters covering {len(universe)} companies, "
      f"{P.shape[1]} of them carrying")
print(f"          a daily price series across {cal[0]:%Y-%m-%d} to {cal[-1]:%Y-%m-%d}")
print(f"Scores:   52-week high = close divided by the highest close of the prior {FORM} trading days;")
print(f"          momentum = price return from {FORM - 1} trading days before the sort date to {SKIP} before it")
print(f"Screens:  complete formation window, traded through the holding month, price at or above "
      f"${FLOOR:.0f},")
print(f"          no formation session beyond +/-{JUMP:.0%}; holding returns winsorised at the 1st/99th percentile")
print(f"Sorts:    {N} monthly holding periods, {out.index[0]:%Y-%m} to {out.index[-1]:%Y-%m}, "
      f"median {int(np.median(counts))} names ranked")
print(f"          (range {min(counts)} to {max(counts)}); equal-weighted top quintile minus bottom quintile")
print()
print("Long-short spread                 Monthly    A year      Vol       t   Hit rate   $1 becomes")
for label, key in [("52-week high, on its own", "pth"),
                   ("52-week high, momentum fixed", "pth_n"),
                   ("12-1 momentum, on its own", "mom"),
                   ("12-1 momentum, 52wk high fixed", "mom_n"),
                   ("52-week high, unwinsorised", "pth_raw")]:
    r = stat(out[key])
    print(f"{label:31s} {r['mean']:8.3%}  {r['ann']:8.2%}  {r['vol']:7.2%}  {r['t']:6.2f}     "
          f"{r['hit']:5.1%}       {r['dollar']:6.2f}")
print()
print("Average next-month return by quintile (1 = lowest score, 5 = highest)")
print("  Quintile                            1      2      3      4      5")
print("  52-week high, on its own      " + "".join(f"{v:6.2%} " for v in QP))
print("  52-week high, momentum fixed  " + "".join(f"{v:6.2%} " for v in GA.mean(0)))
print("  12-1 momentum, on its own     " + "".join(f"{v:6.2%} " for v in QM))
print("  12-1 momentum, 52wk high fixed" + "".join(f"{v:6.2%} " for v in GB.mean(0)))
print(f"  Whole sample {stat(out['univ'])['mean']:.3%} a month "
      f"({stat(out['univ_raw'])['mean']:.3%} before winsorisation)")
print()
print("Momentum quintile first, 52-week-high quintile formed inside it")
print("            high1  high2  high3  high4  high5     5-1")
for i in range(5):
    print(f"  mom{i + 1}    " + "".join(f"{v:6.2%} " for v in GA[i]) + f" {GA[i, 4] - GA[i, 0]:6.2%}")
print()
print("52-week-high quintile first, momentum quintile formed inside it")
print("             mom1   mom2   mom3   mom4   mom5     5-1")
for i in range(5):
    print(f"  high{i + 1}   " + "".join(f"{v:6.2%} " for v in GB[i]) + f" {GB[i, 4] - GB[i, 0]:6.2%}")
print()
print("How far apart are the two rankings?")
print(f"  Mean rank correlation of the two scores  {np.mean(spear):6.3f}   "
      f"({np.min(spear):.3f} to {np.max(spear):.3f})")
print(f"  Top quintile shared by both signals      {np.mean(ovl):6.1%}   "
      f"({np.min(ovl):.1%} to {np.max(ovl):.1%})")
print(f"  Correlation of the two monthly spreads   "
      f"{out[['pth', 'mom']].corr().iloc[0, 1]:6.3f}")

# --------------------------------------------------------------- chart ------

plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
w = np.arange(5)

ax1.bar(w - 0.2, QP * 100, 0.4, color="#f59e0b", label="ranked on its own")
ax1.bar(w + 0.2, GA.mean(0) * 100, 0.4, color="#3b82f6", label="ranked inside momentum quintiles")
ax1.axhline(out["univ"].mean() * 100, color="#e0e0e0", linewidth=0.8, linestyle="--")
ax1.set_xticks(w)
ax1.set_xticklabels(["1 (furthest below)", "2", "3", "4", "5 (at the high)"])
ax1.set_xlabel("Quintile of proximity to the 52-week high; dashed line is the whole sample")
ax1.set_ylabel("Average return, % a month")
ax1.set_title("Proximity to the 52-week high: next-month return by quintile, S&P 500 2016-2026")
ax1.set_ylim(0, 1.45)
ax1.legend(facecolor="#0a0a0a", edgecolor="#3f3f3f", labelcolor="#e0e0e0", loc="upper left",
           framealpha=0.0, ncol=2)

ax2.bar(w, (GB[:, 4] - GB[:, 0]) * 100, 0.55, color="#3b82f6")
ax2.axhline(0, color="#e0e0e0", linewidth=0.8)
ax2.set_xticks(w)
ax2.set_xticklabels(["1 (furthest below)", "2", "3", "4", "5 (at the high)"])
ax2.set_xlabel("Quintile of proximity to the 52-week high")
ax2.set_ylabel("Momentum spread, % a month")
ax2.set_title("Momentum winners minus losers, measured inside each proximity quintile", fontsize=10)
for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(PNG, dpi=150, facecolor="#0a0a0a")

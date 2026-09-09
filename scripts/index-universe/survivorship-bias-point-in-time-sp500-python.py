# Full write-up: https://xfinlink.com/blog/survivorship-bias-point-in-time-sp500-python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = list(range(2005, 2025))
BLOCKS = [("2005-01-01", "2009-12-31"), ("2010-01-01", "2014-12-31"),
          ("2015-01-01", "2019-12-31"), ("2020-01-01", "2024-12-31")]
CHUNK = 50

# --- 1. Two universes: the roster of the day, and the roster of today -------
# Companies are carried by entity id. A ticker points at whoever holds it now,
# so a symbol that changed hands would swap one company for another mid-panel.
pit = {y: {int(x) for x in
           xfl.index("sp500", as_of=f"{y}-01-01", limit=1000)["entity_id"].dropna()}
       for y in YEARS}
current = {int(x) for x in xfl.index("sp500", limit=1000)["entity_id"].dropna()}
ids = sorted(set().union(*pit.values()) | current)

print(f"Distinct companies across the 20 January rosters: {len(set().union(*pit.values()))}")
print(f"Companies on the roster as it stands today:       {len(current)}")
print(f"Of the January 2005 roster ({len(pit[2005])}), still on today's list: "
      f"{len(pit[2005] & current)}")

# --- 2. Monthly total returns for every company either universe ever holds --
frames = []
for start, end in BLOCKS:
    for i in range(0, len(ids), CHUNK):
        frames.append(xfl.prices(entity_id=ids[i:i + CHUNK], start=start, end=end,
                                 interval="1mo", fields=["adj_close", "return_daily"],
                                 max_rows=200000))
px = pd.concat(frames, ignore_index=True)
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
window = (px["m"] >= pd.Period("2005-01")) & (px["m"] <= pd.Period("2024-12"))
px = px[window]
R = px.pivot(index="m", columns="entity_id", values="return_daily").sort_index()
A = px.pivot(index="m", columns="entity_id", values="adj_close").sort_index()

# Integrity screen: the reported monthly return has to agree with the step in
# the company's own split-adjusted price. A company where the two disagree by
# more than 0.5 in logs in any month leaves the panel, both universes alike.
flagged = ((np.log1p(R) - np.log(A).diff()).abs() > 0.5).any()
keep = [c for c in R.columns if not flagged[c]]
print(f"\nPanel: {R.shape[0]} months x {R.shape[1]} companies, "
      f"{int(R.notna().sum().sum()):,} company-months; "
      f"{R.shape[1] - len(keep)} companies set aside by the agreement screen")
R = R[keep]
print(f"Monthly total returns after the screen run "
      f"{R.min().min()*100:.1f}% to {R.max().max()*100:.1f}%")

# --- 3. Two equal-weight books over the same months --------------------------
rows = []
for m in R.index:
    a = R.loc[m, [c for c in R.columns if c in pit[m.year]]].dropna()
    b = R.loc[m, [c for c in R.columns if c in current]].dropna()
    rows.append({"m": m, "pit": a.mean(), "n_pit": len(a),
                 "sur": b.mean(), "n_sur": len(b)})
P = pd.DataFrame(rows).set_index("m")
n = len(P)

ann = {k: (1 + P[k]).prod() ** (12 / n) - 1 for k in ("pit", "sur")}
vol = {k: P[k].std(ddof=1) * np.sqrt(12) for k in ("pit", "sur")}
grow = {k: (1 + P[k]).prod() for k in ("pit", "sur")}

print("\n" + "=" * 78)
print("WHAT A CURRENT MEMBERSHIP LIST ADDS TO A BACKTEST THAT NEVER HAPPENED")
print("=" * 78)
print("Universe   S&P 500 members, monthly total returns, January 2005 - December 2024")
print("Book A     the roster as it stood on 1 January of each year, reset annually")
print("Book B     the roster as the API serves it today, held across all 20 years")
print("Weights    equal, reset every month; a company enters the month it has a")
print("           return and leaves when it stops printing one")
print()
print(f"{'':28s}{'annual':>9s}{'vol':>9s}{'x growth':>11s}{'names':>8s}{'names':>8s}")
print(f"{'':28s}{'':>9s}{'':>9s}{'':>11s}{'2005':>8s}{'2024':>8s}")
for k, label in (("pit", "Point-in-time roster"), ("sur", "Today's roster")):
    print(f"  {label:26s}{ann[k]*100:8.2f}%{vol[k]*100:8.2f}%{grow[k]:10.2f}x"
          f"{P['n_'+k].iloc[0]:8d}{P['n_'+k].iloc[-1]:8d}")
print(f"  {'Difference':26s}{(ann['sur']-ann['pit'])*100:8.2f}pp"
      f"{'':9s}{grow['sur']/grow['pit']:9.2f}x")

by_year = P.groupby(P.index.year).apply(
    lambda d: pd.Series({"pit": (1 + d["pit"]).prod() - 1,
                         "sur": (1 + d["sur"]).prod() - 1}))
by_year["gap"] = (by_year["sur"] - by_year["pit"]) * 100

print("\nCalendar year returns")
print(f"  {'year':6s}{'point-in-time':>15s}{'today':>10s}{'gap (pp)':>11s}")
for y, r in by_year.iterrows():
    print(f"  {y:<6d}{r['pit']*100:14.2f}%{r['sur']*100:9.2f}%{r['gap']:10.2f}")
print(f"\n  Today's roster wins in {int((by_year['gap'] > 0).sum())} of "
      f"{len(by_year)} years; median gap {by_year['gap'].median():.2f}pp")

print("\nFive-year blocks, annualised")
for a, b in ((2005, 2009), (2010, 2014), (2015, 2019), (2020, 2024)):
    d = P[(P.index.year >= a) & (P.index.year <= b)]
    pa = (1 + d["pit"]).prod() ** (12 / len(d)) - 1
    sa = (1 + d["sur"]).prod() ** (12 / len(d)) - 1
    print(f"  {a}-{b}   point-in-time {pa*100:6.2f}%   today {sa*100:6.2f}%   "
          f"gap {(sa-pa)*100:5.2f}pp")

# --- 4. Chart ----------------------------------------------------------------
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])
x = P.index.to_timestamp()
ax1.plot(x, (1 + P["sur"]).cumprod(), color="#3b82f6", lw=2,
         label="Today's membership list, run backwards")
ax1.plot(x, (1 + P["pit"]).cumprod(), color="#e0e0e0", lw=2,
         label="The membership list of the day")
ax1.set_ylabel("Growth of $1")
ax1.set_title("Survivorship bias in an S&P 500 backtest, 2005-2024")
ax1.legend(frameon=False, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)
ax2.bar(by_year.index, by_year["gap"], color="#3b82f6", width=0.7)
ax2.axhline(0, color="#e0e0e0", lw=0.8)
ax2.set_ylabel("Yearly gap\n(percentage points)")
ax2.set_xlabel("Year")
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("survivorship-bias-point-in-time-sp500-python.png", dpi=150,
            facecolor="#0a0a0a")

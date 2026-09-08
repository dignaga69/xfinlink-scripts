# Full write-up: https://xfinlink.com/blog/sell-in-may-by-sector-python
"""
Which sectors actually drive "Sell in May"?

The Halloween indicator says equities earn most of their return between
November and April and little between May and October. This script splits that
claim by sector, using monthly total returns for an equal-weighted
cross-section of point-in-time S&P 500 members, 2001-2024.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup
xfl.set_timeout(300)

SLUG = "sell-in-may-by-sector-python"
START, END = "2001-01-01", "2024-12-31"
YEARS = range(2001, 2025)
WINTER = {11, 12, 1, 2, 3, 4}


def fetch(fn, **kw):
    for attempt in range(4):
        try:
            return fn(**kw)
        except Exception as exc:                      # noqa: BLE001
            print(f"  retry {attempt + 1}: {type(exc).__name__}")
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


# 1. The index as it stood at the start of each year, so the cross-section is
#    never built out of the names that happen to be members today.
rosters = {y: sorted(set(xfl.index("sp500", as_of=f"{y}-01-01", limit=1000)
                         ["entity_id"].dropna().astype(int)))
           for y in YEARS}
universe = sorted(set().union(*[set(v) for v in rosters.values()]))
print(f"point-in-time universe: {len(universe)} entities")

# 2. Monthly total returns, requested by entity id so that a ticker later
#    reassigned to another company cannot contaminate the series.
chunks = [universe[i:i + 25] for i in range(0, len(universe), 25)]
px = pd.concat([fetch(xfl.prices, entity_id=c, start=START, end=END,
                      interval="1mo", fields=["return_daily"], max_rows=300000)
                for c in chunks], ignore_index=True)

px["date"] = pd.to_datetime(px["date"])
px["year"] = px["date"].dt.year
px["month"] = px["date"].dt.to_period("M")

# 3. Keep a company's month only while it was an index member that year.
memb = pd.DataFrame([(y, e) for y, ids in rosters.items() for e in ids],
                    columns=["year", "entity_id"])
panel = (px.merge(memb, on=["year", "entity_id"], how="inner")
         .dropna(subset=["return_daily", "gics_sector"]))

# 4. Winsorise each monthly cross-section at its 1st and 99th percentiles, so
#    one collapsing or rebounding company cannot set a sector's average.
lo = panel.groupby("month")["return_daily"].transform(lambda s: s.quantile(0.01))
hi = panel.groupby("month")["return_daily"].transform(lambda s: s.quantile(0.99))
panel["ret"] = panel["return_daily"].clip(lo, hi)
print(f"panel: {len(panel):,} company-months, {panel['entity_id'].nunique()} "
      f"companies, {panel['month'].nunique()} months")

# 5. Equal-weighted sector return for every month, then split by season.
sm = (panel.groupby(["gics_sector", "month"])
      .agg(ret=("ret", "mean"), n=("entity_id", "nunique")).reset_index())
sm = sm[sm["n"] >= 5]
sm["season"] = np.where(sm["month"].dt.month.isin(WINTER), "winter", "summer")

mkt = panel.groupby("month")["ret"].mean().reset_index()
mkt["season"] = np.where(mkt["month"].dt.month.isin(WINTER), "winter", "summer")


def season_row(g):
    w = g.loc[g["season"] == "winter", "ret"]
    s = g.loc[g["season"] == "summer", "ret"]
    t, p = stats.ttest_ind(w, s, equal_var=False)
    return pd.Series({"nov_apr": 6 * w.mean() * 100,
                      "may_oct": 6 * s.mean() * 100,
                      "gap": 6 * (w.mean() - s.mean()) * 100,
                      "t": t, "p": p, "months": len(w) + len(s)})


tab = (sm.groupby("gics_sector").apply(season_row, include_groups=False)
       .sort_values("gap", ascending=False))

print(f"\n{'Sector':<24}{'Nov-Apr':>9}{'May-Oct':>9}{'Gap':>8}{'t':>7}{'p':>8}{'mths':>7}")
print("-" * 72)
for name, r in tab.iterrows():
    print(f"{name:<24}{r['nov_apr']:>8.2f}%{r['may_oct']:>8.2f}%{r['gap']:>7.2f}"
          f"{r['t']:>7.2f}{r['p']:>8.3f}{r['months']:>7.0f}")
m = season_row(mkt)
print("-" * 72)
print(f"{'All members, equal weight':<24}{m['nov_apr']:>8.2f}%{m['may_oct']:>8.2f}%"
      f"{m['gap']:>7.2f}{m['t']:>7.2f}{m['p']:>8.3f}{m['months']:>7.0f}")

# 6. Chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333"})
order = tab.iloc[::-1]
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(y + 0.21, order["nov_apr"], height=0.42, color="#3b82f6",
        label="November to April")
ax.barh(y - 0.21, order["may_oct"], height=0.42, color="#8a8f98",
        label="May to October")
ax.axvline(0, color="#555555", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(order.index)
ax.set_xlabel("Average six-month return (%)")
ax.set_title("Sell in May by sector: S&P 500 members, 2001-2024")
ax.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
          loc="lower right")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(f"{SLUG}.png", dpi=150, facecolor="#0a0a0a")

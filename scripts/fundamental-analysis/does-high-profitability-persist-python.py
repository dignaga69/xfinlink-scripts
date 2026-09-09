# Full write-up: https://xfinlink.com/blog/does-high-profitability-persist-python
"""Does a company's profitability survive five years, and does its growth rate?

Rank S&P 500 members by operating profitability and by three-year revenue growth,
then measure where the same companies sit five fiscal years later.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

BASE_YEARS = range(2008, 2019)      # formation years
HORIZON = 5                         # fiscal years ahead
FIELDS = ["entity_id", "period_end", "revenue", "operating_income", "total_assets"]

# Point-in-time membership, carried by permanent entity id so that a ticker
# reassigned to another company later cannot substitute one firm for another.
rosters, ids = {}, set()
for y in BASE_YEARS:
    roster = xfl.index("sp500", as_of=f"{y + 1}-01-01")
    rosters[y] = set(roster["entity_id"].dropna().astype(int))
    ids |= rosters[y]
ids = sorted(ids)

frames = []
for i in range(0, len(ids), 50):
    frames.append(xfl.fundamentals(entity_id=ids[i:i + 50], period_type="annual",
                                   start="2004-01-01", end="2024-12-31",
                                   fields=FIELDS, max_rows=200000))
f = pd.concat(frames, ignore_index=True)
f["period_end"] = pd.to_datetime(f["period_end"])

# Label each annual period by the calendar year it mostly covers, taken from the
# period end date rather than from a reported year label, and keep one row per
# company-year.
f["year"] = np.where(f["period_end"].dt.month <= 6,
                     f["period_end"].dt.year - 1, f["period_end"].dt.year)
f = (f.sort_values(["entity_id", "year", "period_end"])
       .drop_duplicates(["entity_id", "year"], keep="last"))

rev = f.pivot_table(index="entity_id", columns="year", values="revenue")
oi = f.pivot_table(index="entity_id", columns="year", values="operating_income")
ta = f.pivot_table(index="entity_id", columns="year", values="total_assets")

# Comparability screen. Two shapes of company-year are not a comparable twelve
# months of trading and leave the sample: one whose revenue falls below half of
# both the year before and the year after, and one reporting operating income
# larger than revenue.
oi = oi.reindex_like(rev)
ta = ta.reindex_like(rev)
neighbour = np.minimum(rev.shift(1, axis=1), rev.shift(-1, axis=1))
drop = (rev <= 0) | ((rev < 0.5 * neighbour) & neighbour.notna()) | (oi > rev)
rev, oi = rev.mask(drop), oi.mask(drop)
usable = (rev > 0) & (ta > 0) & oi.notna()

prof = (oi / ta).where(usable)
growth = pd.DataFrame(index=rev.index, columns=rev.columns, dtype=float)
for y in rev.columns:
    if (y - 3) in rev.columns:
        growth[y] = ((rev[y] / rev[y - 3]) ** (1 / 3) - 1).where(usable[y] & (rev[y - 3] > 0))


def cohorts(mat):
    """Quintile every formation year, then locate the same names HORIZON years on."""
    out = []
    for y in BASE_YEARS:
        if (y + HORIZON) not in mat.columns:
            continue
        members = [i for i in rosters[y] if i in mat.index]
        s = mat.loc[members, [y, y + HORIZON]].dropna()
        s.columns = ["x0", "x5"]
        s["q0"] = pd.qcut(s["x0"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        s["q5"] = pd.qcut(s["x5"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        s["base"] = y
        out.append(s.reset_index())
    return pd.concat(out, ignore_index=True)


def summarise(d):
    t = d.groupby("q0").agg(n=("x0", "size"), now=("x0", "mean"),
                            later=("x5", "mean"), rank5=("q5", "mean"))
    t["stay_top"] = d.assign(v=d["q5"] == 5).groupby("q0")["v"].mean()
    t["to_bottom"] = d.assign(v=d["q5"] == 1).groupby("q0")["v"].mean()
    return t


P, G = cohorts(prof), cohorts(growth)

print(f"Formation years {min(BASE_YEARS)}-{max(BASE_YEARS)}, horizon +{HORIZON} fiscal years")
print(f"Company-years: {len(P):,} profitability, {len(G):,} revenue growth")
print(f"Companies: {P['entity_id'].nunique()} profitability, {G['entity_id'].nunique()} growth\n")

for name, d in [("OPERATING PROFITABILITY  (operating income / total assets)", P),
                ("THREE-YEAR REVENUE GROWTH  (annualised)", G)]:
    t = summarise(d)
    print(name)
    print(f"{'quintile':>9}{'n':>7}{'at formation':>14}{'5 years on':>12}"
          f"{'mean quintile':>15}{'stays top':>11}{'falls bottom':>14}")
    for q in range(1, 6):
        r = t.loc[q]
        print(f"{q:>9}{int(r['n']):>7}{r['now']:>13.1%}{r['later']:>12.1%}"
              f"{r['rank5']:>15.2f}{r['stay_top']:>11.1%}{r['to_bottom']:>14.1%}")
    rho = [spearmanr(x["x0"], x["x5"]).statistic for _, x in d.groupby("base")]
    print(f"  top-minus-bottom spread {t['now'][5] - t['now'][1]:.1%} at formation, "
          f"{t['later'][5] - t['later'][1]:.1%} five years on "
          f"({(t['later'][5] - t['later'][1]) / (t['now'][5] - t['now'][1]):.0%} retained)")
    print(f"  rank correlation across five years: {spearmanr(d['x0'], d['x5']).statistic:.3f} pooled, "
          f"{np.mean(rho):.3f} average by cohort (range {min(rho):.3f} to {max(rho):.3f})\n")

# ── chart ────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#3f3f3f", "font.size": 11})
fig, ax = plt.subplots(figsize=(10, 5))
q = np.arange(1, 6)
ax.axhline(3.0, color="#6b7280", ls="--", lw=1.2)
ax.text(1.05, 3.06, "no persistence: every group averages 3.0", color="#9ca3af", fontsize=9)
ax.plot(q, summarise(P)["rank5"], "o-", color="#3b82f6", lw=2.4, ms=8,
        label="Operating profitability")
ax.plot(q, summarise(G)["rank5"], "s-", color="#f59e0b", lw=2.4, ms=8,
        label="Three-year revenue growth")
ax.set_xticks(q, ["1\nlowest", "2", "3", "4", "5\nhighest"])
ax.set_yticks(np.arange(1, 6))
ax.set_ylim(1, 5)
ax.set_xlabel("Group at the start, one fifth of the index in each")
ax.set_ylabel("Average group five years later")
ax.set_title("Profitability stays put; revenue growth does not")
ax.legend(frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("does-high-profitability-persist-python.png", dpi=150, facecolor="#0a0a0a")

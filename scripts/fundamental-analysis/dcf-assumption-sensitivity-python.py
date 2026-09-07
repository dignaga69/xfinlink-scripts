# Full write-up: https://xfinlink.com/blog/dcf-assumption-sensitivity-python
#
# How much does a DCF depend on its assumptions?
# Current S&P 500 ex financials and real estate, most recent completed fiscal year.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

OUT_PNG = "dcf-assumption-sensitivity-python.png"
DROP_SECTORS = ["Financials", "Real Estate"]
RATES = [0.07, 0.08, 0.09, 0.10, 0.11]      # discount rate grid
TGROW = [0.015, 0.020, 0.025, 0.030]        # terminal growth grid
BASE_R, BASE_G = 0.09, 0.025                # the central cell
YEARS = 10                                  # explicit forecast years
SPAN = 6                                    # fiscal years in the revenue CAGR
G_CAP = 0.12                                # ceiling on the assumed growth rate

# ---------------------------------------------------------------- 1. universe
ids = sorted(int(i) for i in xfl.index("sp500")["entity_id"].dropna())

fun = xfl.fundamentals(entity_id=ids, period_type="annual",
                       start="2019-01-01", end="2026-06-30",
                       fields=["revenue", "free_cash_flow"], max_rows=60000)

cap = []
for i in range(0, len(ids), 60):
    cap.append(xfl.metrics(entity_id=ids[i:i + 60], period_type="annual",
                           start="2025-06-30", end="2026-06-30",
                           fields=["market_cap"], max_rows=20000))
cap = pd.concat(cap, ignore_index=True)

# ------------------------------------------- 2. base cash flow and growth rate
fun = fun.dropna(subset=["gics_sector"])
fun = fun[~fun["gics_sector"].isin(DROP_SECTORS)]
fun = fun.sort_values(["entity_id", "period_end"]).drop_duplicates(
    ["entity_id", "period_end"], keep="last")

rows = []
for eid, g in fun.groupby("entity_id"):
    g = g.sort_values("period_end")
    if len(g) < SPAN + 1:
        continue
    last = g.iloc[-1]
    if last["period_end"] < pd.Timestamp("2025-06-30"):
        continue
    fcf3 = g["free_cash_flow"].iloc[-3:]
    if fcf3.isna().any() or fcf3.mean() <= 0:
        continue
    rev_now, rev_then = g["revenue"].iloc[-1], g["revenue"].iloc[-(SPAN + 1)]
    if pd.isna(rev_now) or pd.isna(rev_then) or rev_then <= 0 or rev_now <= 0:
        continue
    rows.append({"entity_id": eid, "ticker": last["ticker"],
                 "sector": last["gics_sector"], "period_end": last["period_end"],
                 "base_fcf": float(fcf3.mean()), "last_fcf": float(fcf3.iloc[-1]),
                 "g1": float(np.clip((rev_now / rev_then) ** (1 / SPAN) - 1, 0.0, G_CAP))})

d = pd.DataFrame(rows)
n_fund = len(d)

cap = cap.sort_values(["entity_id", "period_end"]).drop_duplicates(
    ["entity_id", "period_end"], keep="last")
d = d.merge(cap[["entity_id", "period_end", "market_cap"]],
            on=["entity_id", "period_end"], how="left")
n_cap = int(d["market_cap"].notna().sum())
d = d[d["market_cap"].between(1_000, 10_000_000)].reset_index(drop=True)

# ------------------------------------------------------- 3. the valuation grid
def dcf(fcf0, g1, r, gt, years=YEARS):
    yrs = np.arange(1, years + 1)
    cf = fcf0 * (1 + g1) ** yrs
    pv = float((cf / (1 + r) ** yrs).sum())
    tv = float(cf[-1] * (1 + gt) / (r - gt) / (1 + r) ** years)
    return pv + tv, tv

grid = [(r, gt) for r in RATES for gt in TGROW]
vals = np.array([[dcf(row.base_fcf, row.g1, r, gt)[0] for r, gt in grid]
                 for row in d.itertuples()])
d["v_min"], d["v_max"] = vals.min(axis=1), vals.max(axis=1)
d["band"] = d["v_max"] / d["v_min"]

base = np.array([dcf(row.base_fcf, row.g1, BASE_R, BASE_G) for row in d.itertuples()])
d["v_base"], d["tv_share"] = base[:, 0], base[:, 1] / base[:, 0] * 100
d["ratio"] = d["market_cap"] / d["v_base"]
d["pos"] = np.where(d["market_cap"] > d["v_max"], "above",
                    np.where(d["market_cap"] < d["v_min"], "below", "inside"))

# base-year choice and horizon, two axes the grid does not cover
d["v_last"] = [dcf(r.last_fcf, r.g1, BASE_R, BASE_G)[0] if r.last_fcf > 0 else np.nan
               for r in d.itertuples()]
sw = (d["v_last"] / d["v_base"] - 1).abs() * 100
d["v_15"] = [dcf(r.base_fcf, r.g1, BASE_R, BASE_G, 15)[0] for r in d.itertuples()]
hz = (d["v_15"] / d["v_base"] - 1) * 100

# ----------------------------------------------------------------- 4. printout
pc = d["pos"].value_counts()
pct = pc / len(d) * 100
print("A ten-year two-stage DCF over 5 discount rates x 4 terminal growth rates")
print("Sample:    current S&P 500 ex financials and real estate; %d companies clear the"
      % n_fund)
print("           cash-flow and growth screens, %d carry a market capitalisation on the"
      % n_cap)
print("           same fiscal year end, %d survive the sanity filter" % len(d))
print("Base:      mean free cash flow of the last 3 fiscal years, %s to %s"
      % (d["period_end"].min().date(), d["period_end"].max().date()))
print("Growth:    own %d-year revenue CAGR, floored at 0%% and capped at %.0f%%; median %.1f%%"
      % (SPAN, G_CAP * 100, d["g1"].median() * 100))
print()
print("How wide is the answer?")
print("  Highest / lowest fair value across the 20 cells")
print("    median %.2fx      quartiles %.2fx to %.2fx      widest %.2fx"
      % (d["band"].median(), d["band"].quantile(.25), d["band"].quantile(.75), d["band"].max()))
print("  Terminal value as a share of the central-cell value")
print("    median %.1f%%      quartiles %.1f%% to %.1f%%"
      % (d["tv_share"].median(), d["tv_share"].quantile(.25), d["tv_share"].quantile(.75)))
print()
print("Band width is a pure function of the growth assumed, not of the company")
print("  Growth      1.0%   2.0%   4.0%   6.0%   8.0%  10.0%  12.0%")
line_b, line_t = [], []
for g1 in [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
    vs = [dcf(100.0, g1, r, gt)[0] for r, gt in grid]
    tot, tv = dcf(100.0, g1, BASE_R, BASE_G)
    line_b.append("%5.2fx" % (max(vs) / min(vs)))
    line_t.append("%5.1f%%" % (tv / tot * 100))
print("  Band width %s" % "  ".join(line_b))
print("  TV share   %s" % "  ".join(line_t))
print()
print("Share of the %d companies the model calls undervalued, cell by cell" % len(d))
print("                          terminal growth")
print("  Discount rate   %s" % "  ".join("%6.1f%%" % (g * 100) for g in TGROW))
cheap = np.zeros((len(RATES), len(TGROW)))
for i, r in enumerate(RATES):
    for j, gt in enumerate(TGROW):
        v = np.array([dcf(row.base_fcf, row.g1, r, gt)[0] for row in d.itertuples()])
        cheap[i, j] = (d["market_cap"].values < v).mean() * 100
    print("  %11.1f%%   %s" % (r * 100, "  ".join("%6.1f%%" % c for c in cheap[i])))
print("  Same companies, same filings: %.1f%% undervalued in the friendliest cell, %.1f%% in the harshest"
      % (cheap.max(), cheap.min()))
print()
print("Can the model tell the market it is wrong?")
print("  above the whole band   %3d   %4.1f%%" % (pc.get("above", 0), pct.get("above", 0)))
print("  inside the band        %3d   %4.1f%%" % (pc.get("inside", 0), pct.get("inside", 0)))
print("  below the whole band   %3d   %4.1f%%" % (pc.get("below", 0), pct.get("below", 0)))
print("  market cap / central-cell value, median %.2f, quartiles %.2f to %.2f"
      % (d["ratio"].median(), d["ratio"].quantile(.25), d["ratio"].quantile(.75)))
print()
print("Two assumptions the grid never varies")
print("  Base year: latest fiscal year free cash flow instead of the 3-year mean")
print("    median absolute change %.1f%%, upper quartile %.1f%%, ninth decile %.1f%%"
      % (sw.median(), sw.quantile(.75), sw.quantile(.90)))
print("    moves value further than the entire rate and terminal grid for %d of %d companies"
      % (int(((d["v_last"] / d["v_base"]).sub(1).abs() > (d["band"] - 1)).sum()), len(d)))
print("  Horizon: 15 explicit years instead of 10")
print("    median change %+.1f%%, quartiles %+.1f%% to %+.1f%%"
      % (hz.median(), hz.quantile(.25), hz.quantile(.75)))
print()
sec = d.groupby("sector").agg(n=("band", "size"), g1=("g1", "median"),
                              band=("band", "median"), tv=("tv_share", "median"))
sec["above"] = d.groupby("sector")["pos"].apply(lambda s: (s == "above").mean() * 100)
sec["inside"] = d.groupby("sector")["pos"].apply(lambda s: (s == "inside").mean() * 100)
sec["g1"] = sec["g1"] * 100
sec = sec.sort_values("band")
print("By sector")
print("  %-24s %4s %7s %8s %9s %8s %8s" % ("", "n", "growth", "band", "TV share", "above", "inside"))
for s, r in sec.iterrows():
    print("  %-24s %4d %6.1f%% %7.2fx %8.1f%% %7.1f%% %7.1f%%"
          % (s, r["n"], r["g1"], r["band"], r["tv"], r["above"], r["inside"]))
print()
print("The ten largest companies in the sample, $bn at their own fiscal year end")
print("  %-7s %-11s %7s %9s %10s %10s %11s %8s"
      % ("Ticker", "Year end", "growth", "base FCF", "low value", "high value",
         "market cap", "verdict"))
for r in d.nlargest(10, "market_cap").itertuples():
    print("  %-7s %-11s %6.1f%% %9.1f %10.0f %10.0f %11.0f %8s"
          % (r.ticker, str(r.period_end.date()), r.g1 * 100, r.base_fcf / 1000,
             r.v_min / 1000, r.v_max / 1000, r.market_cap / 1000, r.pos))

# ------------------------------------------------------------------ 5. chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
                     "axes.edgecolor": "#333333", "font.size": 11})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

cmap = LinearSegmentedColormap.from_list("xfl", ["#0f172a", "#1d4ed8", "#93c5fd"])
ax1.imshow(cheap, cmap=cmap, aspect="auto", vmin=0, vmax=130)
for i in range(len(RATES)):
    for j in range(len(TGROW)):
        ax1.text(j, i, "%.0f%%" % cheap[i, j], ha="center", va="center",
                 color="#e0e0e0", fontsize=11)
ax1.set_xticks(range(len(TGROW)), ["%.1f%%" % (g * 100) for g in TGROW])
ax1.set_yticks(range(len(RATES)), ["%.0f%%" % (r * 100) for r in RATES])
ax1.set_xlabel("Terminal growth rate")
ax1.set_ylabel("Discount rate")
ax1.set_title("Share of the %d companies the model calls undervalued, cell by cell" % len(d),
              fontsize=11.5)
for sp in ax1.spines.values():
    sp.set_visible(False)

share = (d.groupby("sector")["pos"].value_counts(normalize=True)
         .unstack(fill_value=0) * 100).reindex(sec.index[::-1])
for col in ["below", "inside", "above"]:
    if col not in share:
        share[col] = 0.0
left = np.zeros(len(share))
for col, colour, lab in [("below", "#6b7280", "below the band"),
                         ("inside", "#3b82f6", "inside the band"),
                         ("above", "#f59e0b", "above the band")]:
    ax2.barh(share.index, share[col], left=left, color=colour, label=lab, height=0.7)
    left = left + share[col].values
ax2.set_xlabel("Share of companies (%)")
ax2.set_xlim(0, 100)
ax2.set_title("Where the market capitalisation sits against a company's own value band",
              fontsize=11.5, pad=26)
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=9.5)
ax2.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, facecolor="#0a0a0a")

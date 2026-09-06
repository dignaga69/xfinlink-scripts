# Full write-up: https://xfinlink.com/blog/does-company-size-slow-revenue-growth-python
"""Gibrat's law in the large-cap cross-section.

Does a bigger company grow revenue more slowly? Point-in-time S&P 500 rosters
at two start dates, carried by entity id, five-year forward revenue growth
regressed on log size, with and without sector fixed effects.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl
from scipy.stats import spearmanr

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

WINDOWS = [("2014-12-31", 2014, 2019), ("2019-12-31", 2019, 2024)]
IMG = "does-company-size-slow-revenue-growth-python.png"


def panel(as_of, y0, y1):
    """One row per company: base-year revenue and the forward five-year CAGR."""
    roster = xfl.index("sp500", as_of=as_of).drop_duplicates("entity_id")
    ids = sorted(int(e) for e in roster["entity_id"].dropna())
    df = pd.concat(
        [xfl.fundamentals(entity_id=ids[i:i + 40], period_type="all",
                          start=f"{y0 - 1}-06-01", end=f"{y1 + 1}-06-30",
                          fields=["revenue"])
         for i in range(0, len(ids), 40)], ignore_index=True)

    ann = df[df["period_type"] == "annual"].copy()
    ann = ann.sort_values(["entity_id", "fiscal_year", "period_end"])
    ann = ann.drop_duplicates(["entity_id", "fiscal_year"], keep="last")
    ann["year"] = ann["period_end"].dt.year
    ann = ann.drop_duplicates(["entity_id", "year"], keep="last")
    ann = ann[ann["revenue"] > 0]

    # An annual figure must agree with the company's own four quarterly filings
    # for the same year. Where they disagree by more than half, the two are
    # measured on different businesses (a divestiture restatement, typically)
    # and the year cannot anchor a growth rate.
    qtr = df[(df["period_type"] == "quarterly") & df["revenue"].notna()]
    qsum = qtr.groupby(["entity_id", "fiscal_year"])["revenue"].agg(["sum", "size"])
    qsum = qsum[qsum["size"] == 4]["sum"].rename("qsum")
    ann = ann.merge(qsum, left_on=["entity_id", "fiscal_year"], right_index=True, how="left")
    ann = ann[ann["qsum"].isna() | ann["qsum"].between(0.67 * ann["revenue"], 1.5 * ann["revenue"])]

    base = ann[ann["year"] == y0].set_index("entity_id")
    end = ann[ann["year"] == y1].set_index("entity_id")
    m = base.join(end, lsuffix="_b", rsuffix="_f", how="inner")
    m = m[m["gics_sector_b"] != "Financials"]
    m["years"] = (m["period_end_f"] - m["period_end_b"]).dt.days / 365.25
    m = m[m["years"].between(4.0, 6.0)]
    m["cagr"] = (m["revenue_f"] / m["revenue_b"]) ** (1 / m["years"]) - 1
    m["log_size"] = np.log10(m["revenue_b"] * 1e6)
    m["quintile"] = pd.qcut(m["log_size"], 5, labels=[1, 2, 3, 4, 5])
    return len(ids), m


def fits(m):
    """OLS of forward growth on log size, plain and with sector dummies."""
    plain = sm.OLS(m["cagr"], sm.add_constant(m[["log_size"]])).fit()
    dummies = pd.get_dummies(m["gics_sector_b"], drop_first=True, dtype=float)
    sector = sm.OLS(m["cagr"], sm.add_constant(pd.concat([m[["log_size"]], dummies], axis=1))).fit()
    return plain, sector


results = {}
for as_of, y0, y1 in WINDOWS:
    n_roster, m = panel(as_of, y0, y1)
    results[(y0, y1)] = (n_roster, m, *fits(m))

print("=" * 74)
print("DOES COMPANY SIZE SLOW REVENUE GROWTH?")
print("Point-in-time S&P 500 rosters ex Financials, five-year forward revenue CAGR")
print("=" * 74)
print(f"{'window':<16}{'roster':>8}{'in sample':>12}{'median revenue':>17}{'median CAGR':>14}")
for (y0, y1), (n_roster, m, _, _) in results.items():
    print(f"{f'{y0} to {y1}':<16}{n_roster:>8}{len(m):>12}"
          f"{f'${m.revenue_b.median() / 1000:,.1f}bn':>17}{m.cagr.median() * 100:>13.1f}%")

print("\nREVENUE CAGR REGRESSED ON log10(BASE REVENUE)")
print(f"{'window':<16}{'model':<22}{'slope per 10x':>15}{'t':>8}{'p':>9}{'R2':>8}")
for (y0, y1), (_, m, plain, sector) in results.items():
    for label, r in (("size only", plain), ("+ sector fixed effects", sector)):
        print(f"{f'{y0} to {y1}':<16}{label:<22}{r.params['log_size'] * 100:>13.2f}pp"
              f"{r.tvalues['log_size']:>8.2f}{r.pvalues['log_size']:>9.4f}{r.rsquared:>8.3f}")

print("\nRANK CORRELATION BETWEEN SIZE AND FORWARD GROWTH (Spearman)")
for (y0, y1), (_, m, _, _) in results.items():
    rho, p = spearmanr(m["log_size"], m["cagr"])
    within = m.groupby("gics_sector_b", observed=True).apply(
        lambda g: spearmanr(g["log_size"], g["cagr"])[0] if len(g) > 9 else np.nan,
        include_groups=False)
    print(f"{f'{y0} to {y1}':<16}all companies rho {rho:+.3f} (p={p:.4f})"
          f"   median within sector rho {within.median():+.3f}")

print("\nMEAN FORWARD REVENUE CAGR BY SIZE QUINTILE (Q1 smallest)")
print(f"{'window':<16}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'Q5':>9}{'Q1 - Q5':>10}")
for (y0, y1), (_, m, _, _) in results.items():
    q = m.groupby("quintile", observed=True)["cagr"].mean() * 100
    print(f"{f'{y0} to {y1}':<16}" + "".join(f"{v:>8.1f}%" for v in q) + f"{q[1] - q[5]:>9.1f}pp")

print("\nMEDIAN BASE REVENUE BY SIZE QUINTILE, $bn")
print(f"{'window':<16}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'Q5':>9}")
for (y0, y1), (_, m, _, _) in results.items():
    q = m.groupby("quintile", observed=True)["revenue_b"].median() / 1000
    print(f"{f'{y0} to {y1}':<16}" + "".join(f"{v:>9.1f}" for v in q))

print("\nSPREAD OF FIVE-YEAR CAGR WITHIN EACH SIZE QUINTILE (standard deviation, pp)")
print(f"{'window':<16}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'Q5':>9}")
for (y0, y1), (_, m, _, _) in results.items():
    q = m.groupby("quintile", observed=True)["cagr"].std() * 100
    print(f"{f'{y0} to {y1}':<16}" + "".join(f"{v:>9.1f}" for v in q))

# ── chart ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
    "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#333333", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

n_roster, m, plain, _ = results[(2019, 2024)]
ax1.scatter(m["revenue_b"] / 1000, m["cagr"] * 100, s=14, color="#3b82f6", alpha=0.55,
            edgecolors="none")
grid = np.linspace(m["log_size"].min(), m["log_size"].max(), 50)
ax1.plot(10 ** grid / 1e9, (plain.params["const"] + plain.params["log_size"] * grid) * 100,
         color="#f59e0b", lw=2)
ax1.set_xscale("log")
ax1.set_xlabel("Revenue in 2019 ($bn, log scale)")
ax1.set_ylabel("Revenue growth 2019-2024 (% a year)")
ax1.set_title("Bigger companies, slightly slower growth", fontsize=11, pad=8)
ax1.axhline(0, color="#555555", lw=0.8)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

width = 0.38
x = np.arange(5)
for i, ((y0, y1), (_, mm, _, _)) in enumerate(results.items()):
    q = mm.groupby("quintile", observed=True)["cagr"].mean() * 100
    ax2.bar(x + (i - 0.5) * width, q.values, width,
            color=["#3b82f6", "#93c5fd"][i], label=f"{y0}-{y1}")
ax2.set_xticks(x, ["Q1\nsmallest", "Q2", "Q3", "Q4", "Q5\nlargest"])
ax2.set_ylabel("Mean revenue growth (% a year)")
ax2.set_xlabel("Size quintile by revenue at the start of the window")
ax2.set_title("The drag is not a straight line", fontsize=11, pad=8)
ax2.axhline(0, color="#555555", lw=0.8)
ax2.legend(frameon=False, fontsize=9)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig(IMG, dpi=150, facecolor="#0a0a0a")
print(f"\nchart saved to {IMG}")

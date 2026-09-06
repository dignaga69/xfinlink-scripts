# Full write-up: https://xfinlink.com/blog/foreign-revenue-dollar-beta-python
#
# Does a company's foreign revenue share make its stock dollar-sensitive?
# Foreign revenue share comes from reported geographic segment revenue.
# Dollar sensitivity is the coefficient on a major-currency dollar index in a
# two-factor daily return regression that also carries the market.
# Universe: the S&P 500 as it stood on 2023-09-01, carried by entity id.

import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xfinlink as xfl

warnings.filterwarnings("ignore")

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

AS_OF, START, END = "2023-09-01", "2023-08-25", "2026-08-31"

# Dollar index built from single-currency funds, weighted like the dollar index.
CCY = {"FXE": 0.60, "FXY": 0.14, "FXB": 0.12, "FXC": 0.10, "FXF": 0.04}

# Labels that mean the United States and nothing else.
DOM = {("united", "states"), ("u", "s"), ("us",), ("usa",), ("domestic",),
       ("united", "states", "of", "america"), ("united", "states", "and", "territories")}

print("=== Foreign Revenue Share and Firm-Level Dollar Beta ===")

# ---- universe -----------------------------------------------------------
roster = xfl.index("sp500", as_of=AS_OF)
ids = sorted(roster["entity_id"].unique().tolist())
print(f"S&P 500 roster as of {AS_OF}: {len(ids)} entities")

# ---- foreign revenue share from reported geographic segments ------------
parts = []
for i in range(0, len(ids), 50):
    d = xfl.fundamentals(entity_id=ids[i:i + 50], include_segments=True,
                         period_type="annual", start="2023-09-01", end="2026-09-06")
    if len(d):
        parts.append(d)
fund = pd.concat(parts, ignore_index=True)
fund = fund[fund["segments_geographic"].apply(lambda v: isinstance(v, list) and len(v) > 0)]
fund = fund.sort_values("period_end").groupby("entity_id", as_index=False).last()

recs = []
for _, r in fund.iterrows():
    segs = [s for s in r["segments_geographic"]
            if s.get("is_primary") and s.get("value") is not None]
    total = sum(s["value"] for s in segs)
    dom = [s["value"] for s in segs
           if tuple(re.sub(r"[^a-z ]", " ", s["label"].lower()).split()) in DOM]
    if len(dom) != 1 or not total or not (0.98 <= total / r["revenue"] <= 1.02):
        continue
    recs.append(dict(entity_id=r["entity_id"], ticker=r["ticker"], sector=r["gics_sector"],
                     foreign_share=1 - dom[0] / total))
seg = pd.DataFrame(recs)
print(f"Companies reporting a United States revenue line that reconciles "
      f"to total revenue: {len(seg)}")

# ---- dollar index and market -------------------------------------------
mp = xfl.prices(list(CCY) + ["SPY"], start=START, end=END, fields=["adj_close"])
mp = mp.pivot_table(index="date", columns="ticker", values="adj_close").dropna()
mr = mp.pct_change(fill_method=None).dropna()
mr["USD"] = -sum(w * mr[t] for t, w in CCY.items())
print(f"Dollar index: {len(mr)} daily observations, "
      f"{mr['USD'].std() * np.sqrt(252):.1%} annualised volatility, "
      f"cumulative {(1 + mr['USD']).prod() - 1:+.1%}")

# ---- stock returns ------------------------------------------------------
sids = sorted(seg["entity_id"].tolist())
px = pd.concat([xfl.prices(entity_id=sids[i:i + 40], start=START, end=END,
                           fields=["adj_close"])
                for i in range(0, len(sids), 40)], ignore_index=True)
wide = px.pivot_table(index="date", columns="ticker", values="adj_close").reindex(mr.index)
rets = wide.pct_change(fill_method=None)

keep, short, extreme = [], 0, 0
for t in rets.columns:
    s = rets[t]
    if s.notna().sum() < 0.95 * len(mr):
        short += 1
    elif s.max() > 1.0 or s.min() < -0.5:
        extreme += 1
    else:
        keep.append(t)
print(f"Names without a complete price history for the window: {short}; "
      f"outside the return screen: {extreme}")

# ---- dollar beta, with and without the market control -------------------
X2 = sm.add_constant(mr[["SPY", "USD"]])
X1 = sm.add_constant(mr[["USD"]])
rows = []
for t in keep:
    y = rets[t]
    m = y.notna()
    f2 = sm.OLS(y[m], X2[m]).fit()
    f1 = sm.OLS(y[m], X1[m]).fit()
    rows.append(dict(ticker=t, mkt_beta=f2.params["SPY"], usd_beta=f2.params["USD"],
                     usd_t=f2.tvalues["USD"], raw_usd_beta=f1.params["USD"],
                     n_obs=int(m.sum())))
df = seg.merge(pd.DataFrame(rows), on="ticker").dropna(subset=["usd_beta"])

print(f"\nFinal sample: {len(df)} companies, "
      f"{df['n_obs'].min()}-{df['n_obs'].max()} daily observations each")
print(f"Dollar beta: mean {df['usd_beta'].mean():+.3f}, "
      f"median {df['usd_beta'].median():+.3f}, sd {df['usd_beta'].std():.3f}")
print(f"Negative dollar beta: {(df['usd_beta'] < 0).sum()} of {len(df)} companies; "
      f"individually significant at 5 percent: {(df['usd_t'].abs() > 1.96).sum()}")
print(f"Foreign revenue share: mean {df['foreign_share'].mean():.1%}, "
      f"range {df['foreign_share'].min():.1%} to {df['foreign_share'].max():.1%}")

# ---- cross-sectional test ----------------------------------------------
D = pd.get_dummies(df["sector"], drop_first=True, dtype=float)
specs = [
    ("dollar beta ~ foreign share", "usd_beta", df[["foreign_share"]]),
    ("+ sector fixed effects", "usd_beta", pd.concat([df[["foreign_share"]], D], axis=1)),
    ("no market control", "raw_usd_beta", df[["foreign_share"]]),
    ("market beta ~ foreign share", "mkt_beta", df[["foreign_share"]]),
]
print("\n%-30s %9s %8s %8s %7s" % ("specification", "coef", "t-stat", "p", "R2"))
for label, dep, Z in specs:
    f = sm.OLS(df[dep], sm.add_constant(Z)).fit()
    print("%-30s %9.4f %8.2f %8.3f %7.3f" % (label, f.params["foreign_share"],
          f.tvalues["foreign_share"], f.pvalues["foreign_share"], f.rsquared))

print(f"\nPearson correlation {df['usd_beta'].corr(df['foreign_share']):+.3f}, "
      f"Spearman {df['usd_beta'].corr(df['foreign_share'], method='spearman'):+.3f}")

df["q"] = pd.qcut(df["foreign_share"], 5, labels=[1, 2, 3, 4, 5])
qt = df.groupby("q", observed=True).agg(n=("ticker", "size"),
                                        foreign=("foreign_share", "mean"),
                                        usd_beta=("usd_beta", "mean"),
                                        mkt_beta=("mkt_beta", "mean"))
print("\nquintile   n  foreign share  dollar beta  market beta")
for q, r in qt.iterrows():
    print(f"{q:5d} {r['n']:6.0f} {r['foreign']:14.1%} "
          f"{r['usd_beta']:12.3f} {r['mkt_beta']:12.2f}")

lo, hi = df[df["q"] == 1]["usd_beta"], df[df["q"] == 5]["usd_beta"]
t_stat, p_val, _ = sm.stats.ttest_ind(hi, lo, usevar="unequal")
print(f"\nQ5 minus Q1 dollar beta: {hi.mean() - lo.mean():+.3f}  "
      f"t = {t_stat:.2f}  p = {p_val:.3f}")

# ---- chart --------------------------------------------------------------
BG, FG, ACC = "#0a0a0a", "#e0e0e0", "#3b82f6"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#333333")

ax1.scatter(df["foreign_share"] * 100, df["usd_beta"], s=22, color=ACC, alpha=0.75)
xs = np.linspace(0, df["foreign_share"].max() * 100, 50)
fit = sm.OLS(df["usd_beta"], sm.add_constant(df[["foreign_share"]])).fit()
ax1.plot(xs, fit.params["const"] + fit.params["foreign_share"] * xs / 100,
         color="#f59e0b", lw=2)
ax1.axhline(0, color="#555555", lw=0.8)
ax1.set_xlabel("Share of revenue earned outside the United States (%)", color=FG)
ax1.set_ylabel("Sensitivity to a rising dollar", color=FG)
ax1.set_title("Does foreign revenue make a stock dollar-sensitive?", color=FG, fontsize=13)

w, idx = 0.38, np.arange(len(qt))
ax2.bar(idx - w / 2, qt["usd_beta"], w, color=ACC, label="Dollar sensitivity")
ax2.bar(idx + w / 2, qt["mkt_beta"], w, color="#f59e0b", label="Market beta")
ax2.axhline(0, color="#555555", lw=0.8)
ax2.set_xticks(idx)
ax2.set_xticklabels([f"Q{q}\n{v:.0%} foreign" for q, v in zip(qt.index, qt["foreign"])],
                    color=FG)
ax2.set_ylabel("Beta", color=FG)
ax2.set_title("By foreign revenue quintile: market beta rises, dollar sensitivity does not",
              color=FG, fontsize=13)
leg = ax2.legend(facecolor=BG, edgecolor="#333333", labelcolor=FG)
plt.tight_layout()
plt.savefig("foreign-revenue-dollar-beta-python.png", dpi=150, facecolor=BG)
print("\nChart saved to foreign-revenue-dollar-beta-python.png")

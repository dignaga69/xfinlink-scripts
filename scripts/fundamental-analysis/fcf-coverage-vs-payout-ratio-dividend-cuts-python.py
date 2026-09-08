# Full write-up: https://xfinlink.com/blog/fcf-coverage-vs-payout-ratio-dividend-cuts-python
#
# Which coverage measure warns of a dividend cut first: the earnings payout
# ratio, or free cash flow coverage? Point-in-time S&P 500 roster as at
# 28 June 2019, predictors from fiscal years already closed and filed, dividend
# outcomes read from ex-dividend records over the following 18 months.

import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl
from scipy.stats import fisher_exact

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

FORM = pd.Timestamp("2019-06-28")       # formation date
BASE_START = pd.Timestamp("2018-06-29")  # baseline dividend year opens
WIN_END = pd.Timestamp("2020-12-31")     # outcome window closes

FIELDS = ["net_income", "free_cash_flow", "operating_cash_flow",
          "capital_expenditures", "dividends_paid_common"]


def retry(fn, **kw):
    for attempt in range(4):
        try:
            return fn(**kw)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def batches(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


# ---------------------------------------------------------------- universe
roster = xfl.index("sp500", as_of="2019-06-28")
ids = sorted(roster["entity_id"].dropna().astype(int).unique())

# Fiscal years closing on or before 31 Dec 2018 are public well before the
# formation date, so nothing here is known after the fact.
with ThreadPoolExecutor(8) as ex:
    fund = pd.concat(ex.map(
        lambda b: retry(xfl.fundamentals, entity_id=b, period_type="annual",
                        start="2017-06-01", end="2018-12-31", fields=FIELDS,
                        max_rows=5000),
        batches(ids, 40)), ignore_index=True)

fund = (fund.sort_values(["entity_id", "period_end"])
            .groupby("entity_id", as_index=False).tail(1))
payers = fund[fund["dividends_paid_common"] > 0].copy()

# ------------------------------------------------- dividends actually paid
with ThreadPoolExecutor(8) as ex:
    px = pd.concat(ex.map(
        lambda b: retry(xfl.prices, entity_id=b, start="2018-06-01",
                        end="2020-12-31", fields=["dividend", "split_ratio"],
                        max_rows=5000),
        batches(payers["entity_id"].astype(int).tolist(), 6)), ignore_index=True)

# a declared dividend is stated in the shares of its day, so divide out every
# split that lands after the ex-date to put the whole series on one basis
px = px.sort_values(["entity_id", "date"])
px["sr"] = pd.to_numeric(px["split_ratio"], errors="coerce").fillna(1.0)
px["fwd"] = px.groupby("entity_id")["sr"].transform(
    lambda s: s[::-1].cumprod()[::-1].shift(-1).fillna(1.0))
px["div"] = pd.to_numeric(px["dividend"], errors="coerce") / px["fwd"]

# companies whose listing ends inside the window were bought, not stressed
alive = px.groupby("entity_id")["date"].max()
alive = set(alive[alive >= WIN_END - pd.Timedelta(days=15)].index)
divs = px[(px["div"] > 0) & px["entity_id"].isin(alive)][["entity_id", "date", "div"]]

rows = []
for eid, g in divs.groupby("entity_id"):
    g = g.sort_values("date")
    base = g[(g["date"] > BASE_START) & (g["date"] <= FORM)]
    if len(base) < 4:                      # not a regular payer at formation
        continue
    rate = base["div"].median()            # median shrugs off a special dividend
    gap = base["date"].diff().dt.days.dropna().median()
    post = g[(g["date"] > FORM) & (g["date"] <= WIN_END)]
    reduced = bool((post["div"] < 0.90 * rate).any())
    seq = pd.concat([base["date"].tail(1), post["date"], pd.Series([WIN_END])])
    omitted = bool(seq.diff().dt.days.max() > 1.75 * gap)
    rows.append({"entity_id": eid, "base_rate": rate, "n_base": len(base),
                 "n_post": len(post), "post_min": post["div"].min() if len(post) else np.nan,
                 "reduced": reduced, "omitted": omitted, "cut": reduced or omitted})

df = payers.merge(pd.DataFrame(rows), on="entity_id", how="inner")
df = df[df["net_income"].notna() & df["free_cash_flow"].notna()].copy()
df["payout"] = df["dividends_paid_common"] / df["net_income"]
df["cover"] = df["free_cash_flow"] / df["dividends_paid_common"]
df["earn_flag"] = (df["net_income"] <= 0) | (df["payout"] > 0.80)
df["cash_flag"] = df["cover"] < 1.0
n, base_rate = len(df), df["cut"].mean()

# ------------------------------------------------------------------ output
W = 84
print("=" * W)
print("DIVIDEND CUTS: EARNINGS PAYOUT RATIO VERSUS FREE CASH FLOW COVERAGE")
print("=" * W)
print(f"Roster     S&P 500 as at 2019-06-28, carried by entity id: {len(ids)} companies")
print(f"Sample     {n} of them were paying a regular dividend at that date and")
print( "           had a closed fiscal year on file")
print( "Predictors last fiscal year ending on or before 2018-12-31, so every")
print( "           figure was public six months before the window opens")
print( "Outcome    a cut is a per-share payment below 90% of the pre-formation")
print( "           median, or a gap of more than 1.75 normal payment intervals,")
print(f"           between 2019-06-28 and {WIN_END.date()}")
print()
print(f"Cuts in the window: {int(df.cut.sum())} of {n} companies "
      f"({base_rate * 100:.1f}%)  |  outright reductions {int(df.reduced.sum())}, "
      f"skipped payments {int(df.omitted.sum())}")
print()

print("Each screen on its own")
print(f"  {'screen':26s}{'flagged':>9s}{'cut rate':>10s}{'not flagged':>13s}{'cut rate':>10s}")
for label, m in [("Earnings payout above 80%", df.earn_flag),
                 ("FCF coverage below 1.0x", df.cash_flag)]:
    a, b = df[m], df[~m]
    print(f"  {label:26s}{len(a):>9d}{a.cut.mean() * 100:>9.1f}%{len(b):>13d}"
          f"{b.cut.mean() * 100:>9.1f}%")
print()

print("The two screens crossed")
print(f"  {'':28s}{'companies':>11s}{'cuts':>7s}{'cut rate':>10s}")
cells = [("Neither flag", ~df.earn_flag & ~df.cash_flag),
         ("Cash flow flag only", ~df.earn_flag & df.cash_flag),
         ("Earnings flag only", df.earn_flag & ~df.cash_flag),
         ("Both flags", df.earn_flag & df.cash_flag)]
for label, m in cells:
    s = df[m]
    print(f"  {label:28s}{len(s):>11d}{int(s.cut.sum()):>7d}{s.cut.mean() * 100:>9.1f}%")
both, neither = df[df.earn_flag & df.cash_flag], df[~df.earn_flag & ~df.cash_flag]
tab = [[int(both.cut.sum()), len(both) - int(both.cut.sum())],
       [int(neither.cut.sum()), len(neither) - int(neither.cut.sum())]]
print(f"  Both flags against neither, Fisher exact p = {fisher_exact(tab)[1]:.5f}")
print()

df["cov_b"] = pd.cut(df["cover"], [-np.inf, 0, 1, 1.5, 2.5, np.inf],
                     labels=["negative FCF", "0.0 to 1.0x", "1.0 to 1.5x",
                             "1.5 to 2.5x", "above 2.5x"])
df["pay_b"] = pd.cut(df["payout"].where(df["net_income"] > 0, -1),
                     [-np.inf, 0, 0.4, 0.6, 0.8, np.inf],
                     labels=["loss-making", "0 to 40%", "40 to 60%", "60 to 80%",
                             "above 80%"])
for col, head in [("cov_b", "Cut rate by free cash flow coverage of the dividend"),
                  ("pay_b", "Cut rate by earnings payout ratio")]:
    t = df.groupby(col, observed=True).agg(n=("cut", "size"), cuts=("cut", "sum"),
                                           rate=("cut", "mean"))
    print(head)
    print(f"  {'bucket':16s}{'n':>6s}{'cuts':>7s}{'rate':>9s}")
    for k, r in t.iterrows():
        print(f"  {str(k):16s}{int(r['n']):>6d}{int(r['cuts']):>7d}{r['rate'] * 100:>8.1f}%")
    print()

print("By sector")
sec = df.groupby("gics_sector").agg(n=("cut", "size"), cuts=("cut", "sum"),
                                    rate=("cut", "mean"), mcov=("cover", "median"),
                                    mpay=("payout", "median")).sort_values("rate", ascending=False)
print(f"  {'sector':24s}{'n':>5s}{'cuts':>6s}{'rate':>9s}{'med cover':>11s}{'med payout':>12s}")
for k, r in sec.iterrows():
    print(f"  {k:24s}{int(r['n']):>5d}{int(r['cuts']):>6d}{r['rate'] * 100:>8.1f}%"
          f"{r['mcov']:>10.2f}x{r['mpay'] * 100:>11.0f}%")
print()

flagged = df.earn_flag | df.cash_flag
missed = df.cut & ~flagged
print(f"Cuts that neither screen flagged: {int(missed.sum())} of {int(df.cut.sum())} "
      f"({missed.sum() / df.cut.sum() * 100:.1f}%)")
cd = df[(df.gics_sector == "Consumer Discretionary") & ~flagged]
print(f"Consumer Discretionary names clean on both screens: {len(cd)}, of which "
      f"{int(cd.cut.sum())} cut ({cd.cut.mean() * 100:.1f}%)")
print("=" * W)

# ------------------------------------------------------------------- chart
plt.rcParams.update({"figure.facecolor": "#0a0a0a", "axes.facecolor": "#0a0a0a",
                     "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
                     "xtick.color": "#e0e0e0", "ytick.color": "#e0e0e0"})
fig, ax = plt.subplots(figsize=(10, 5))
labels = [c[0] for c in cells]
rates = [df[c[1]].cut.mean() * 100 for c in cells]
counts = [int(df[c[1]].cut.sum()) for c in cells]
sizes = [int(c[1].sum()) for c in cells]
bars = ax.bar(labels, rates, color="#3b82f6", width=0.6)
ax.axhline(base_rate * 100, color="#e0e0e0", linestyle="--", linewidth=1)
ax.text(-0.42, base_rate * 100 + 1.0, f"all {n} companies: {base_rate * 100:.1f}%",
        ha="left", fontsize=9, color="#e0e0e0")
for b, r, c, s in zip(bars, rates, counts, sizes):
    x = b.get_x() + b.get_width() / 2
    ax.text(x, r - 2.4, f"{r:.1f}%", ha="center", fontsize=12, color="#ffffff")
    ax.text(x, r - 5.0, f"{c} of {s}", ha="center", fontsize=9, color="#dbeafe")
ax.set_ylabel("Companies that cut the dividend within 18 months (%)")
ax.set_title("Dividend cuts after June 2019, by what the coverage screens flagged")
ax.set_ylim(0, max(rates) * 1.12)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("bottom", "left"):
    ax.spines[side].set_color("#333333")
plt.tight_layout()
plt.savefig("fcf-coverage-vs-payout-ratio-dividend-cuts-python.png", dpi=150,
            facecolor="#0a0a0a")

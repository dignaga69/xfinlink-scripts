# Full write-up: https://xfinlink.com/blog/negative-book-equity-sp500-python
"""Does negative book equity signal distress?

Screens the current S&P 500 roster for companies whose most recent annual
balance sheet reports total equity below zero, then checks each one against
its own earnings and free cash flow for the same fiscal year.
"""

import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

PNG = "negative-book-equity-sp500-python.png"
FIELDS = ["revenue", "net_income", "total_equity", "retained_earnings",
          "treasury_stock", "total_assets", "operating_cash_flow",
          "capital_expenditures", "gics_sector"]

roster = xfl.index("sp500")
ids = sorted(int(i) for i in roster["entity_id"].dropna().unique())

f = pd.concat([xfl.fundamentals(entity_id=ids[i:i + 50], period_type="annual",
                                fields=FIELDS, start="2024-01-01",
                                end="2026-09-10", max_rows=50000)
               for i in range(0, len(ids), 50)], ignore_index=True)

# One row per company: the most recent annual balance sheet, fiscal 2025 or later
bs = f[f["total_equity"].notna()].sort_values("period_end")
latest = bs.groupby("entity_id").tail(1)
s = latest[latest["period_end"] >= "2025-01-01"].copy()
s["fcf"] = s["operating_cash_flow"] - s["capital_expenditures"]

neg = s[s["total_equity"] < 0].sort_values("total_equity")

print(f"current S&P 500 roster: {len(roster)} members")
print(f"members with an annual balance sheet for fiscal 2025 or later: {len(s)}")
print(f"of those, reporting negative book equity: {len(neg)} "
      f"({100 * len(neg) / len(s):.1f}%)\n")

print(f"{'':6s}{'sector':24s}{'FY end':12s}{'revenue':>10s}{'equity':>10s}"
      f"{'net inc':>9s}{'FCF':>9s}")
for _, r in neg.iterrows():
    print(f"{r['ticker']:6s}{r['gics_sector'][:23]:24s}"
          f"{r['period_end']:%Y-%m-%d}  {r['revenue']:10,.0f}"
          f"{r['total_equity']:10,.0f}{r['net_income']:9,.0f}{r['fcf']:9,.0f}")

print(f"\ncombined book equity {neg['total_equity'].sum():>12,.0f}")
print(f"combined net income  {neg['net_income'].sum():>12,.0f}")
print(f"combined free cash flow {neg['fcf'].sum():>9,.0f}")
print(f"profitable in that fiscal year: {(neg['net_income'] > 0).sum()} of {len(neg)}")
print(f"free cash flow positive:        {(neg['fcf'] > 0).sum()} of {len(neg)}")
print(f"median return on equity as reported: "
      f"{100 * (neg['net_income'] / neg['total_equity']).median():.0f}%")
print("\nsectors: " + ", ".join(f"{k} {v}" for k, v
                                in neg["gics_sector"].value_counts().items()))

mcd = neg[neg["ticker"] == "MCD"].iloc[0]
low = neg[neg["ticker"] == "LOW"].iloc[0]
print(f"\nMCD  retained earnings {mcd['retained_earnings']:>9,.0f}   "
      f"treasury stock {mcd['treasury_stock']:>9,.0f}   equity {mcd['total_equity']:>8,.0f}")
print(f"LOW  retained earnings {low['retained_earnings']:>9,.0f}   "
      f"{'':27s}equity {low['total_equity']:>8,.0f}")

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")

y = range(len(neg))
ax.barh([i + 0.2 for i in y], neg["total_equity"] / 1000, height=0.4,
        color="#9ca3af", label="Book equity")
ax.barh([i - 0.2 for i in y], neg["fcf"] / 1000, height=0.4,
        color="#3b82f6", label="Free cash flow, same year")
ax.set_yticks(list(y))
ax.set_yticklabels(neg["ticker"], color="#e0e0e0", fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color="#e0e0e0", lw=0.8)
ax.set_xlabel("$ billions", color="#e0e0e0")
ax.set_title("Negative book equity against annual free cash flow, S&P 500",
             color="#e0e0e0")
ax.tick_params(colors="#e0e0e0")
ax.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0",
          loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(PNG, dpi=150, facecolor="#0a0a0a")
print(f"\nchart written to {PNG}")

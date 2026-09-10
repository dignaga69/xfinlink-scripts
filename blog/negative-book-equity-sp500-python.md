**Does Negative Book Equity Signal Distress? S&P 500 Balance Sheet Screening in Python**

September 10, 2026 · BALANCE-SHEET-HEALTH

**What's the question?**

Book equity is what remains after a company's liabilities are subtracted from its assets. When the figure is below zero, the balance sheet states that the company owes more than it owns. That is the textbook definition of accounting insolvency, and it breaks two widely used screening ratios: price-to-book gains a negative denominator, and return on equity flips sign while the business is still earning money.

Twenty-nine S&P 500 companies carrying a current annual balance sheet report negative book equity. Either the index holds a pocket of insolvent companies, or book equity is measuring something other than solvency. Six per cent of a sample is too large to handle by accident.

**The approach**

The test sets one accounting figure against two cash figures from the same fiscal year.

1. Universe: the current S&P 500 roster, retrieved through the index endpoint and carried by entity id rather than by ticker, so each company is followed by its own identity.
2. For each company, take the most recent annual balance sheet whose fiscal year ends on 1 January 2025 or later. That window holds 474 of the 504 members, and those 474 form the sample.
3. Flag every company whose reported total equity is below zero.
4. Against each flagged company set net income and free cash flow, defined as operating cash flow minus capital expenditure, for that same fiscal year.

Step 4 is the actual test: insolvency has a cash meaning as well as an accounting one, and a company that cannot meet its obligations reveals it in the cash flow statement.

**Code**

```python
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

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

print(f"negative book equity: {len(neg)} of {len(s)}")
print(f"profitable: {(neg['net_income'] > 0).sum()}   "
      f"free cash flow positive: {(neg['fcf'] > 0).sum()}")
print(f"combined equity {neg['total_equity'].sum():,.0f}   "
      f"combined free cash flow {neg['fcf'].sum():,.0f}")
```

Full script with formatting and visualisation: [negative-book-equity-sp500-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/balance-sheet-health/negative-book-equity-sp500-python.py)

**Output**

```
current S&P 500 roster: 504 members
members with an annual balance sheet for fiscal 2025 or later: 474
of those, reporting negative book equity: 29 (6.1%)

      sector                  FY end         revenue    equity  net inc      FCF
PM    Consumer Staples        2025-12-31      40,648    -9,994   11,348   10,664
LOW   Consumer Discretionary  2026-01-30      86,286    -9,917    6,654    7,651
TDG   Industrials             2025-09-30       8,831    -9,686    2,074    1,816
SBUX  Consumer Discretionary  2025-09-28      37,184    -8,097    1,856    2,442
YUM   Consumer Discretionary  2025-12-31       8,214    -7,325    1,559    1,639
HCA   Health Care             2025-12-31      75,600    -6,027    6,784    7,692
BKNG  Consumer Discretionary  2025-12-31      26,917    -5,578    5,404    9,087
OTIS  Industrials             2025-12-31      14,431    -5,392    1,384    1,444
HLT   Consumer Discretionary  2025-12-31      12,039    -5,388    1,457    2,028
SBAC  Real Estate             2025-12-31       2,815    -4,854    1,054        1
DPZ   Consumer Discretionary  2025-12-28       4,940    -3,901      602      672
MAR   Consumer Discretionary  2025-12-31      26,186    -3,771    2,601    2,608
MO    Consumer Staples        2025-12-31      23,279    -3,502    6,947    9,074
AZO   Consumer Discretionary  2025-08-30      18,939    -3,414    2,498    1,790
ABBV  Health Care             2025-12-31      61,160    -3,270    4,226   17,816
CAH   Health Care             2026-06-30     254,248    -2,883    1,714    4,525
MSCI  Financials              2025-12-31       3,134    -2,655    1,202    1,549
DELL  Information Technology  2026-01-30     113,538    -2,470    5,936    8,552
MCK   Health Care             2026-03-31     403,430    -2,172    4,762    5,719
VRSN  Information Technology  2025-12-31       1,657    -2,154      826    1,068
MCD   Consumer Discretionary  2025-12-31      26,885    -1,791    8,563   10,197
FICO  Information Technology  2025-09-30       1,991    -1,746      652      770
IRM   Real Estate             2025-12-31       6,902      -981      152     -932
ORLY  Consumer Discretionary  2025-12-31      17,782      -763    2,538    1,593
DVA   Health Care             2025-12-31      13,643      -651      747    1,311
HPQ   Information Technology  2025-10-31      55,295      -346    2,529    2,800
WYNN  Consumer Discretionary  2025-12-31       7,138      -275      327      692
MAS   Industrials             2025-12-31       7,562      -185      810      866
MTD   Health Care             2025-12-31       4,026       -24      869      849

combined book equity     -109,212
combined net income        88,076
combined free cash flow   115,983
profitable in that fiscal year: 29 of 29
free cash flow positive:        28 of 29
median return on equity as reported: -73%

sectors: Consumer Discretionary 11, Health Care 6, Information Technology 4, Industrials 3, Consumer Staples 2, Real Estate 2, Financials 1

MCD  retained earnings    70,282   treasury stock    79,316   equity   -1,791
LOW  retained earnings   -10,839                              equity   -9,917
```

**What this tells us**

All 29 companies earned a profit in the fiscal year their balance sheets cover, and 28 produced positive free cash flow. The group generated 115,983 million dollars of free cash flow against a combined book deficit of 109,212 million, so one year of cash generation exceeds the entire accumulated shortfall in book value. No distressed cohort produces that pattern.

Shareholder payouts are the mechanism, and the buyback half of them reaches the balance sheet by two routes. McDonald's has retained 70,282 million of past profit and spent 79,316 million buying its own shares, which sits on the balance sheet as treasury stock and is subtracted from equity; the excess of the second figure over the first is most of the reason its equity reads -1,791 million. Lowe's cancels the shares it repurchases instead of holding them, so the cost is charged against retained earnings, which stand at -10,839 million after years of buybacks larger than reported profit. Under either route the deficit records cash handed to shareholders rather than losses.

The sector pattern follows from the mechanism. Eleven of the 29 sit in Consumer Discretionary, where franchised restaurants and specialty retailers pair modest asset bases with steady cash generation, the profile that sustains years of large payouts. Two names read differently: Iron Mountain is the only one with negative free cash flow, at -932 million, and SBA Communications converted 2,815 million of revenue into 1 million of it. Both are real estate companies whose construction spending absorbs nearly all of the operating cash they produce, so their deficits have a different history behind them.

Return on equity computed on these balance sheets has a median of -73 per cent, and McDonald's reports 8,563 million of net income on -1,791 million of equity, which is -478 per cent. The ratio stays valid arithmetically and empty economically.

**So what?**

Treat the sign of book equity as an accounting record of past distributions, not as a solvency signal. Route these companies away from book-based metrics: rank them on enterprise value against operating income or free cash flow, where the denominator survives a buyback, and judge leverage by net debt against cash flow rather than against a book figure driven through zero by repurchases.

Then check what an existing screen does with them. A price-to-book sort that drops negative denominators silently removes 6 per cent of the sample, McDonald's and Booking Holdings included, and the names it removes are companies that returned a lot of cash rather than companies that are expensive. A return-on-equity filter behaves worse, because it keeps them and ranks them at the bottom while they earn 88,076 million between them.

Distress, where it exists, shows up in the cash columns instead. The screen worth running is negative book equity together with negative free cash flow, which flags one company here rather than 29, and that one is capital-intensive real estate rather than a failing business.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*

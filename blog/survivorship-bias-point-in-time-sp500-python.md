**How Much Does Survivorship Bias Add to a Backtest? Point-in-Time S&P 500 Returns in Python**

September 9, 2026 · INDEX-UNIVERSE

**What's the question?**

A backtest needs a universe, and the cheapest one to build is the index membership list as it stands today. That list is not the index of 2005. It omits every company that failed, was taken over, or shrank far enough to be dropped, and those companies left for reasons written into their returns.

Survivorship bias is the name for the error that follows. Naming it is easy; sizing it is the useful part, because the correction is not free. A point-in-time membership record, which stores who belonged to the index on any given date rather than only who belongs now, is a heavier thing to carry than one current list. The measurement below holds everything constant except membership: two equal-weight portfolios, the same twenty years of monthly total returns, the same index, differing only in which companies are eligible in each month.

**The approach**

1. For each year from 2005 to 2024, take the S&P 500 as it stood on 1 January of that year. That is Book A, reset every January.
2. Take the roster as the API serves it today and run it across the same twenty years. That is Book B, the list a backtester reaches for first.
3. Pull monthly total returns for every company either book ever holds, January 2005 to December 2024. Hold each book in equal weight, resetting weights monthly; a company enters the month it prints a return and leaves when it stops printing one.
4. Require agreement between two independent fields: a company-month whose reported return does not match the step in that company's own split-adjusted price, by more than 0.5 in logs, leaves the panel. That removes 61 of 169,341 company-months, from both books alike.
5. Compound both books and compare, by year and over the full window.

Companies are carried by entity id rather than by ticker. A ticker points at whoever holds it now, so a symbol reassigned after a bankruptcy would swap one company for another in the middle of the panel, which is the failure this exercise is meant to measure rather than commit.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

YEARS = list(range(2005, 2025))
BLOCKS = [("2005-01-01", "2009-12-31"), ("2010-01-01", "2014-12-31"),
          ("2015-01-01", "2019-12-31"), ("2020-01-01", "2024-12-31")]

pit = {y: {int(x) for x in
           xfl.index("sp500", as_of=f"{y}-01-01", limit=1000)["entity_id"].dropna()}
       for y in YEARS}
current = {int(x) for x in xfl.index("sp500", limit=1000)["entity_id"].dropna()}
ids = sorted(set().union(*pit.values()) | current)

frames = []
for start, end in BLOCKS:
    for i in range(0, len(ids), 50):
        frames.append(xfl.prices(entity_id=ids[i:i + 50], start=start, end=end,
                                 interval="1mo", fields=["adj_close", "return_daily"],
                                 max_rows=200000))
px = pd.concat(frames, ignore_index=True)
px["m"] = px["date"].dt.to_period("M")
px = px.drop_duplicates(["entity_id", "m"])
R = px.pivot(index="m", columns="entity_id", values="return_daily").sort_index()
A = px.pivot(index="m", columns="entity_id", values="adj_close").sort_index()
R = R.mask((np.log1p(R) - np.log(A).diff()).abs() > 0.5)

rows = []
for m in R.index:
    a = R.loc[m, [c for c in R.columns if c in pit[m.year]]].dropna()
    b = R.loc[m, [c for c in R.columns if c in current]].dropna()
    rows.append({"m": m, "pit": a.mean(), "sur": b.mean()})
P = pd.DataFrame(rows).set_index("m")

for k in ("pit", "sur"):
    print(k, (1 + P[k]).prod() ** (12 / len(P)) - 1, (1 + P[k]).prod())
```

Full script with formatting and visualisation: [survivorship-bias-point-in-time-sp500-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/index-universe/survivorship-bias-point-in-time-sp500-python.py)

**Output**

```
==============================================================================
WHAT A CURRENT MEMBERSHIP LIST ADDS TO A BACKTEST THAT NEVER HAPPENED
==============================================================================
Universe   S&P 500 members, monthly total returns, January 2005 - December 2024
Book A     the roster as it stood on 1 January of each year, reset annually
Book B     the roster as the API serves it today, held across all 20 years
Weights    equal, reset every month; a company enters the month it has a
           return and leaves when it stops printing one

Companies across the 20 January rosters   901
Companies on the roster today             504
January 2005 members still on it today    234 of 498
Panel   240 months x 910 companies, 169,280 company-months, 61 set aside by the agreement screen
Range   1625% (GME, 2021-01) to -99.2% (LEH, 2008-09)

                               annual      vol   x growth   names   names
                                                             2005    2024
  Point-in-time roster          9.97%   17.53%      6.69x     466     496
  Today's roster               15.42%   16.94%     17.60x     404     502
  Difference                    5.45pp              2.63x

Calendar year returns
  year    point-in-time     today   gap (pp)
  2005            7.47%    14.20%      6.74
  2006           16.18%    22.28%      6.10
  2007            0.72%    11.48%     10.76
  2008          -40.05%   -33.37%      6.68
  2009           48.44%    46.74%     -1.69
  2010           21.60%    25.67%      4.07
  2011           -0.01%     3.77%      3.78
  2012           17.23%    22.83%      5.60
  2013           36.27%    41.07%      4.80
  2014           13.97%    18.59%      4.62
  2015           -2.59%     3.74%      6.33
  2016           15.57%    19.41%      3.84
  2017           18.59%    27.11%      8.52
  2018           -7.73%    -2.64%      5.09
  2019           30.00%    34.48%      4.48
  2020           12.91%    20.66%      7.75
  2021           29.69%    31.94%      2.25
  2022          -10.96%   -10.55%      0.41
  2023           14.50%    23.21%      8.71
  2024           12.89%    19.26%      6.38

  Today's roster wins in 19 of 20 years; median gap 5.35pp

Five-year blocks, annualised
  2005-2009   point-in-time   2.28%   today   8.77%   gap  6.49pp
  2010-2014   point-in-time  17.23%   today  21.79%   gap  4.57pp
  2015-2019   point-in-time   9.87%   today  15.57%   gap  5.70pp
  2020-2024   point-in-time  11.00%   today  15.91%   gap  4.91pp

Robustness: clipping each month's cross-section at the 1st and 99th percentiles
  gives 9.86% against 14.94%, a gap of 5.07pp.
```

**What this tells us**

The point-in-time book compounds at 9.97% a year. The same rule run on today's membership list compounds at 15.42%, which is 5.45 percentage points a year of return no investor could have collected, and on a dollar the two books end twenty years apart at 6.69x and 17.60x. Book A sits close to what an equal-weight S&P 500 has actually delivered over a comparable span, near 10% a year, which is the check that matters.

Two mechanisms produce the gap, both the same act of conditioning on an outcome. The first is familiar: companies that failed are absent from today's list, so their losses never enter Book B. Lehman Brothers sat in the index through 2008 and prints -99.2% that September, the worst month in the panel, and it belongs to Book A alone. The second is quieter and does more work. A company that entered the index in 2015 entered because it had already grown, and Book B credits it with every earlier month it has a price for, so the selection runs backwards across the whole panel.

Turnover sets the scale: 901 distinct companies pass through the twenty January rosters to fill about 500 slots, and of the 498 members in January 2005, 234 remain on today's list. The distortion is stable rather than episodic. Today's list wins in 19 of 20 calendar years, each five-year block gives between 4.57 and 6.49 points, and clipping the extreme 2% of each monthly cross-section still leaves 5.07 points. Only 2009 favours the point-in-time book, when the rebound rewarded distressed names today's list no longer contains.

**So what?**

Five percentage points a year is larger than almost any documented factor premium. A momentum screen or a quality tilt tested on a current membership list therefore starts with more free return than the effect it is trying to demonstrate, and the measured edge can be entirely the universe. Ask which list built a backtest before asking anything about its signal.

Two habits fix it. Build the universe at each rebalance date from the roster of that date, so the portfolio holds what was holdable. Then carry each holding by a persistent company identifier rather than by ticker, because a delisted company's symbol is frequently reassigned and joining on symbol quietly replaces the failure with its successor's returns. The first habit removes the bias; the second stops it creeping back in through the join.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*

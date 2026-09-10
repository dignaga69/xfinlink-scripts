**Does Proximity to the 52-Week High Beat Momentum? Conditional Quintile Sorts in Python**

September 10, 2026 · SIGNAL-EVALUATION

**What's the question?**

George and Hwang turned chart-watching folklore into a ranking in 2004: divide the current price by the highest price of the trailing twelve months, buy the names sitting on their 52-week high, sell the names furthest below it. The story behind it is anchoring: buyers hesitate to pay more than a stock has fetched at any point in the past year, so good news is absorbed slowly and the drift runs on.

The ranking is not independent of momentum, which scores stocks on their own past return over the same window. A stock on its high has almost always risen over the past year, so a test of one ranking is partly a test of the other. Does proximity to the high predict next month's return once past return is held fixed, and does momentum still pay among the stocks already sitting on their high?

**The approach**

The sample is the S&P 500 as it actually stood at 127 month ends between January 2016 and July 2026, covering 703 companies. Membership comes from point-in-time rosters and names are carried by entity id rather than ticker, so a symbol that changed hands does not swap one company for another mid-test. Returns are price returns computed from split-adjusted closes.

1. At each month end, keep the roster names with a complete daily series over the 252 trading days ending on the sort date and through the holding month, priced at $5 or more, with no session in that window beyond plus or minus 40%.
2. Proximity score: the close divided by the highest close of the prior 252 trading days. A value of 1.00 means the stock is sitting on its high.
3. Momentum score: price return from 251 trading days before the sort date to 21 before it, the standard 12-1 construction that skips the most recent month.
4. Sort into quintiles on each score, hold equal-weighted for a month, record top quintile minus bottom quintile.
5. Sort conditionally: momentum quintiles first with proximity ranked inside each, then the reverse. Each pass measures one score with the other held fixed.

Holding-month returns are winsorised at the 1st and 99th percentile each month, so no single name carries a quintile average.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

ANCHORS = pd.date_range("2016-01-31", "2026-07-31", freq="ME").strftime("%Y-%m-%d").tolist()
FORM, SKIP, FLOOR, JUMP = 252, 21, 5.0, 0.40

rosters = {a: sorted(int(e) for e in xfl.index("sp500", as_of=a)["entity_id"].dropna())
           for a in ANCHORS}
px = pd.concat([xfl.prices(entity_id=e, start="2014-12-01", end="2026-08-31",
                           fields=["adj_close"], max_rows=200000)
                for e in sorted({e for ids in rosters.values() for e in ids})],
               ignore_index=True).drop_duplicates(["entity_id", "date"])

P = px.pivot(index="date", columns="entity_id", values="adj_close").sort_index()
cal, V = P.index, P.values
ret = np.vstack([np.full((1, V.shape[1]), np.nan), V[1:] / V[:-1] - 1.0])
live = (~np.isnan(V)).cumsum(0)                  # cumulative traded days
jump = (np.abs(ret) > JUMP).cumsum(0)            # cumulative outsized sessions
runmax = pd.DataFrame(V).rolling(FORM, min_periods=FORM).max().values
pos = {e: i for i, e in enumerate(P.columns)}
me = pd.Series(np.arange(len(cal))).groupby(np.asarray(cal.year * 100 + cal.month)).max()
days = [int(me[int(a[:4]) * 100 + int(a[5:7])]) for a in ANCHORS]

quint = lambda x, m: np.argsort(np.argsort(x, kind="stable"), kind="stable") * 5 // m
rows, grids = [], []
for anchor, t, nxt in zip(ANCHORS, days, days[1:]):
    f, s = t - FORM + 1, t - SKIP            # formation start, skip point
    ids = np.array([pos[e] for e in rosters[anchor] if e in pos])
    ids = ids[(live[t, ids] - live[f - 1, ids] == t - f + 1)   # complete formation window
              & (live[nxt, ids] - live[t, ids] == nxt - t)     # traded through the holding month
              & (V[t, ids] >= FLOOR)
              & (jump[t, ids] - jump[f - 1, ids] == 0)]
    n = len(ids)

    pth = V[t, ids] / runmax[t, ids]     # proximity to the 52-week high, 1.0 = sitting on it
    mom = V[s, ids] / V[f, ids] - 1.0    # 12-1 price momentum
    fwd = V[nxt, ids] / V[t, ids] - 1.0  # the month held
    fw = np.clip(fwd, *np.percentile(fwd, [1, 99]))
    qp, qm = quint(pth, n), quint(mom, n)

    B = np.zeros((5, 5))                 # rows 52-week-high quintile, cols momentum inside it
    for i in range(5):
        idx = np.where(qp == i)[0]
        inner = quint(mom[idx], len(idx))
        for j in range(5):
            B[i, j] = fw[idx[inner == j]].mean()

    rows.append((fw[qp == 4].mean() - fw[qp == 0].mean(),   # 52-week high, on its own
                 fw[qm == 4].mean() - fw[qm == 0].mean(),   # momentum, on its own
                 (B[:, 4] - B[:, 0]).mean()))               # momentum, 52-week high held fixed
    grids.append(B)

print(np.array(rows).mean(0))            # monthly long-short means
print(np.array(grids).mean(0))           # momentum inside 52-week-high quintiles
```

Full script with formatting and visualisation: [52-week-high-vs-momentum-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/signal-evaluation/52-week-high-vs-momentum-python.py)

**Output**

```
Proximity to the 52-week high against 12-1 momentum, point-in-time S&P 500
Panel:    127 month-end rosters covering 703 companies, 699 of them carrying
          a daily price series across 2014-12-01 to 2026-08-31
Scores:   52-week high = close divided by the highest close of the prior 252 trading days;
          momentum = price return from 251 trading days before the sort date to 21 before it
Screens:  complete formation window, traded through the holding month, price at or above $5,
          no formation session beyond +/-40%; holding returns winsorised at the 1st/99th percentile
Sorts:    126 monthly holding periods, 2016-02 to 2026-07, median 491 names ranked
          (range 460 to 498); equal-weighted top quintile minus bottom quintile

Long-short spread                 Monthly    A year      Vol       t   Hit rate   $1 becomes
52-week high, on its own         -0.189%    -3.79%   17.87%   -0.41     50.8%         0.67
52-week high, momentum fixed     -0.295%    -4.26%   12.70%   -0.90     48.4%         0.63
12-1 momentum, on its own         0.133%     0.20%   16.76%    0.31     52.4%         1.02
12-1 momentum, 52wk high fixed    0.396%     4.31%   10.38%    1.48     54.0%         1.56
52-week high, unwinsorised       -0.147%    -3.41%   18.49%   -0.31     50.8%         0.69

Average next-month return by quintile (1 = lowest score, 5 = highest)
  Quintile                            1      2      3      4      5
  52-week high, on its own       1.03%  0.90%  0.95%  1.00%  0.84% 
  52-week high, momentum fixed   1.06%  1.01%  0.94%  0.92%  0.77% 
  12-1 momentum, on its own      0.89%  0.94%  0.98%  0.89%  1.02% 
  12-1 momentum, 52wk high fixed 0.76%  1.00%  0.92%  0.88%  1.16% 
  Whole sample 0.943% a month (0.974% before winsorisation)

Momentum quintile first, 52-week-high quintile formed inside it
            high1  high2  high3  high4  high5     5-1
  mom1     1.15%  0.87%  0.93%  0.78%  0.68%  -0.47%
  mom2     1.27%  0.85%  0.78%  0.82%  0.96%  -0.32%
  mom3     0.85%  0.88%  1.17%  1.10%  0.92%   0.07%
  mom4     0.80%  1.03%  0.92%  0.97%  0.74%  -0.06%
  mom5     1.25%  1.40%  0.92%  0.94%  0.56%  -0.69%

52-week-high quintile first, momentum quintile formed inside it
             mom1   mom2   mom3   mom4   mom5     5-1
  high1    0.92%  1.09%  1.00%  0.96%  1.18%   0.26%
  high2    0.73%  0.90%  0.72%  0.86%  1.30%   0.56%
  high3    0.68%  0.94%  0.84%  0.95%  1.35%   0.67%
  high4    0.64%  1.09%  1.24%  0.93%  1.10%   0.47%
  high5    0.84%  0.99%  0.81%  0.69%  0.86%   0.02%

How far apart are the two rankings?
  Mean rank correlation of the two scores   0.607   (-0.079 to 0.886)
  Top quintile shared by both signals       36.1%   (10.2% to 63.6%)
  Correlation of the two monthly spreads    0.837
```

**What this tells us**

The two rankings overlap without being the same. Rank correlation averaged 0.607 across the 126 sorts and swung from -0.079 to 0.886, the top quintiles shared 36.1% of their names, and the monthly long-short returns correlate 0.837.

Neither signal worked on its own. Proximity put 0.84% a month in the top quintile against 1.03% in the bottom, and one dollar in that book ended at 0.67; momentum alone returned 0.133% a month, or 0.20% a year compounded, against a sample average of 0.943%.

Holding one score fixed separates them. Ranked inside momentum quintiles, proximity falls monotonically: 1.06%, 1.01%, 0.94%, 0.92% and 0.77% a month from the fifth furthest below the high to the fifth sitting on it. Ranked inside proximity quintiles, momentum rises to 0.396% a month, or 4.31% a year, at 10.38% volatility and a 54.0% hit rate. Each score does better once the other is held still, in opposite directions.

The second grid locates momentum. Its spread inside the proximity quintiles reads 0.26%, 0.56%, 0.67%, 0.47% and 0.02% a month, so momentum paid in the four quintiles below the high and nothing among the names sitting on it. Inside the top momentum quintile, the fifth of winners furthest below their high returned 1.25% a month against 0.56% for the fifth closest to it.

No t-statistic in the table reaches 2, so no single spread is distinguishable from zero. What holds up is the ordering, consistent across both grids and 126 sorts, with a sign that contradicts the anchoring story here.

**So what?**

Proximity to the 52-week high is not a drop-in replacement for momentum among large caps; ranking on it cost 4.26% a year against momentum-matched peers. The useful version is the reverse: rank on momentum, then use distance below the high as a second dimension and buy the winners that have not yet regained their old high. That cell returned 1.40% a month, the best in either grid.

A screen for stocks within a few percent of the 52-week high inherits most of a momentum book at a rank correlation of 0.607, so its backtest belongs next to momentum rather than next to the index. The conditional sort that settles it costs two rankings and a groupby.

One boundary. This is the S&P 500, where every name is liquid and heavily covered, so slow diffusion of news is least likely here. George and Hwang measured the full cross-section including small caps, and the negative sign here describes large caps.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*

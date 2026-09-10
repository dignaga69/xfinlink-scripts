**How Late Does a Bear Market Signal Arrive? Turning-Point Detection in Python**

September 10, 2026 · MARKET-REGIMES

**What's the question?**

A bear market is defined after the fact: a fall of 20% or more from a previous high. The label attaches on the day the arithmetic is satisfied. What matters to anyone acting on it is how much of the damage has already happened by the time it fires.

The same question applies to trend rules, the most watched of which is the first close below the 200-day moving average, the average closing price over the previous 200 sessions. It fires earlier by construction, because a price has to fall much less to break an average that trails behind it. Earlier is only better if what remains of the decline is worth avoiding, and if the rule does not keep firing in markets that never fall.

Drawdown here means the percentage fall from a running maximum of closing prices.

**The approach**

Six funds cover US large caps (SPY, DIA), mid caps (MDY), small caps (IWM), developed markets outside the US (EFA) and emerging markets (EEM). Each series runs from January 1996 or the fund's launch, whichever is later, to 9 September 2026 on split-adjusted closes, so every figure is a price return and excludes distributions.

1. Track each fund's running maximum and record every spell in which the price fell 20% or more below it before recovering to that peak. The trough is the lowest close inside the spell.
2. Require 250 sessions of history before a peak counts, so the 200-day average exists throughout every decline studied.
3. Within each decline, find the first close 20% below the peak and the first close below the 200-day average after it.
4. At each signal, measure trading days since the peak, the share of the eventual decline already realised, and the further fall to the trough.
5. Count every downside crossing of the 200-day average in the same sample, with the worst close over the six months after each, to price the false alarms.

Both thresholds are conventions fixed in advance rather than values fitted to the sample, and each signal uses only prices available on the day it fires.

**Code**

```python
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup


def drawdown_episodes(s, thresh=0.20):
    """Peak-to-trough declines of at least thresh percent from a running maximum."""
    out, vals = [], s.values
    peak_i = trough_i = 0
    peak_v = trough_v = vals[0]
    in_ep = False
    for i in range(1, len(vals)):
        v = vals[i]
        if v >= peak_v:
            if in_ep:
                out.append((peak_i, trough_i, peak_v, trough_v))
                in_ep = False
            peak_i, trough_i, peak_v, trough_v = i, i, v, v
        else:
            if v < trough_v:
                trough_i, trough_v = i, v
            if not in_ep and v <= peak_v * (1 - thresh):
                in_ep = True
    return out


rows = []
for t in ["SPY", "DIA", "MDY", "IWM", "EFA", "EEM"]:
    px = xfl.prices(t, start="1990-01-01", end="2026-09-09", fields=["adj_close"])
    s = px.sort_values("date").set_index("date")["adj_close"].astype(float)
    ma = s.rolling(200).mean()
    below = (s < ma) & ma.notna()
    cross = below & ~below.shift(1, fill_value=False).astype(bool)

    for peak_i, trough_i, peak_v, trough_v in drawdown_episodes(s):
        if peak_i < 250:      # the 200-day average does not exist yet
            continue
        hits = np.where(cross.values[peak_i + 1:trough_i + 1])[0]
        if not len(hits):
            continue
        r20_i = peak_i + int(np.argmax(s.values[peak_i:trough_i + 1] <= peak_v * 0.8))
        ma_i = peak_i + 1 + int(hits[0])
        depth = trough_v / peak_v - 1
        rows.append({"ticker": t, "decline": depth,
                     "r20_days": r20_i - peak_i,
                     "r20_done": (s.values[r20_i] / peak_v - 1) / depth,
                     "ma_days": ma_i - peak_i,
                     "ma_done": (s.values[ma_i] / peak_v - 1) / depth})

ep = pd.DataFrame(rows)
print(len(ep), ep["decline"].median())
print(ep[["r20_days", "r20_done", "ma_days", "ma_done"]].median())
```

Full script with formatting and visualisation: [how-late-is-a-bear-market-signal-python.py](https://github.com/xfinlink/xfinlink-examples/blob/main/scripts/price-analysis/how-late-is-a-bear-market-signal-python.py)

**Output**

![Trading days from the peak to each signal against the share of the eventual decline already realised, for 27 declines of 20% or more in six index funds](/blog-images/how-late-is-a-bear-market-signal-python.png)

```
=== How late does a bear-market signal arrive? ===
Six index funds, split-adjusted closes to 2026-09-09 (price returns, distributions excluded)
  SPY from 1996-01-02  DIA from 1998-01-20  MDY from 1996-01-02  IWM from 2000-05-26  EFA from 2001-08-17  EEM from 2003-04-11
27 declines of 20% or more from a running peak, 27 of them since recovered

       peak       trough      decline       --- 20% label ---        --- 200-day average ---  
fund                                     days   done  left to fall   days   done  left to fall
MDY    1998-04-22 1998-10-08  -28.1%      89    74%         -9.3%     37    36%        -20.0%
DIA    2000-01-14 2002-10-09  -37.8%     298    54%        -21.8%      9    23%        -31.9%
SPY    2000-03-24 2002-10-09  -49.1%     242    47%        -33.9%     15    20%        -43.6%
MDY    2000-09-07 2002-10-09  -32.4%     143    65%        -14.3%     25    37%        -23.3%
EEM    2004-04-12 2004-05-17  -21.4%      25   100%          0.0%     20    85%         -4.0%
EEM    2006-05-09 2006-06-13  -26.2%      21    79%         -7.1%     20    73%         -8.7%
MDY    2007-06-04 2009-03-09  -56.3%     326    36%        -45.3%     43    16%        -52.0%
IWM    2007-07-09 2009-03-09  -59.9%     134    35%        -49.3%     13    14%        -56.2%
DIA    2007-10-09 2009-03-09  -53.8%     184    39%        -41.6%     23    15%        -49.8%
SPY    2007-10-09 2009-03-09  -56.5%     186    36%        -45.5%     21    10%        -54.0%
EEM    2007-10-31 2008-11-20  -67.2%      53    31%        -58.8%     52    27%        -59.8%
EFA    2007-10-31 2009-03-09  -63.2%      68    33%        -53.6%     13    13%        -60.0%
IWM    2011-04-29 2011-10-03  -29.4%      69    84%         -6.3%     64    28%        -23.0%
MDY    2011-04-29 2011-10-03  -26.5%      69    89%         -4.0%     65    39%        -18.0%
IWM    2015-06-23 2016-02-11  -26.5%     141    84%         -5.6%     23    25%        -21.4%
MDY    2018-08-29 2018-12-24  -23.7%      79    91%         -2.7%     29    30%        -17.8%
IWM    2018-08-31 2020-03-23  -42.3%      73    49%        -27.1%     27    23%        -36.2%
SPY    2018-09-20 2018-12-24  -20.2%      65   100%          0.0%     15    36%        -13.9%
DIA    2020-02-12 2020-03-23  -37.1%      19    55%        -21.1%      8    23%        -31.2%
SPY    2020-02-19 2020-03-23  -34.1%      16    78%        -10.1%      6    35%        -25.1%
MDY    2020-02-20 2020-03-23  -42.5%      12    53%        -25.7%      3    17%        -37.9%
EEM    2021-02-17 2022-10-24  -41.5%     261    49%        -26.4%    109    23%        -35.4%
IWM    2021-11-08 2023-10-27  -33.1%      55    64%        -15.3%     13    25%        -27.2%
MDY    2021-11-16 2022-09-26  -24.4%     143    86%         -4.2%     10    35%        -17.3%
SPY    2022-01-03 2022-10-12  -25.4%     111    85%         -4.9%     13    33%        -18.6%
DIA    2022-01-04 2022-09-30  -21.9%     182    93%         -1.9%     11    25%        -17.3%
MDY    2024-11-25 2025-04-08  -24.5%      88    90%         -3.2%     59    36%        -17.2%

Medians across the 27 declines
  peak-to-trough decline -33.1%, trough reached 195 trading days after the peak
  20% label      fires    88 days after the peak    65% of the decline already done   -14.3% still to fall
  200-day cross  fires    20 days after the peak    25% of the decline already done   -25.1% still to fall

200-day average crossed from above 680 times across the six funds
  27 of those crossings opened one of the declines above (4.0%)
  worst close over the next six months, median -7.1%, 39.8% fell a further 10% or more
```

**What this tells us**

The label arrives four months late. The median gap between the peak and the first close 20% below it was 88 trading days, and at that point 65% of the eventual decline had already happened, leaving 14.3% still to fall against a median decline of 33.1%.

How much is left depends on how deep the decline turns out to be, since 20% is a fixed distance from the peak and the trough is not. In the six declines that ended at the 2008 or 2009 lows, the label attached with 31% to 39% of the loss realised. In the mild ones it landed on the floor: SPY in December 2018 and EEM in May 2004 first closed 20% down on the trough day itself, and DIA in September 2022 reached the mark with 93% of its decline finished. A fixed threshold is least useless exactly where the losses are largest.

The 200-day crossing was earlier in all 27 declines, a median 20 trading days after the peak with 25% realised and 25.1% still to fall. Speed does not always separate the two rules: in the 2008 emerging market decline, EEM broke its average one session before it hit the 20% mark, the fall being steep enough that the average never got ahead of it.

The cost sits in the crossing count. The average was crossed from above 680 times across the six funds, and 27 of those crossings, 4.0%, opened one of the declines above. Among the 674 with a full six months of subsequent data, the median worst close was 7.1% below the crossing price and 39.8% were followed by a fall of 10% or more.

**So what?**

Waiting for the bear-market label is a decision to accept the first two thirds of a decline in full. That is defensible for a long-horizon allocator who wants no timing rule at all, and poor for anyone who treats the label as risk management: the median case sells after 65% of the damage is done.

A 200-day rule is early enough to matter and noisy enough to need sizing. Roughly 24 crossings led nowhere for every one that opened a 20% decline, so a binary exit on each crossing produces a history dominated by whipsaws, while cutting exposure by a fixed fraction turns the same signal into a cost that scales with how often it is wrong. The 39.8% follow-through rate is the number to size against.

Score any timing rule the way these two are scored here: by the share of the decline still ahead of it on the day it fired, not by whether it eventually named the decline.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*

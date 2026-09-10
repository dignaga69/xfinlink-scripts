# Full write-up: https://xfinlink.com/blog/how-late-is-a-bear-market-signal-python
"""How late does a bear-market signal arrive?

Dates every 20%-or-deeper decline in six broad index funds, then measures how
much of each decline was already finished when two real-time rules fired: the
20% bear-market label itself, and the first close below the 200-day average.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

TICKERS = ["SPY", "DIA", "MDY", "IWM", "EFA", "EEM"]
END = "2026-09-09"
MIN_HISTORY = 250  # bars of history required before a peak counts


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
                out.append((peak_i, trough_i, peak_v, trough_v, True))
                in_ep = False
            peak_i, trough_i, peak_v, trough_v = i, i, v, v
        else:
            if v < trough_v:
                trough_i, trough_v = i, v
            if not in_ep and v <= peak_v * (1 - thresh):
                in_ep = True
    if in_ep:
        out.append((peak_i, trough_i, peak_v, trough_v, False))
    return out


rows, crossings, starts = [], [], {}
for t in TICKERS:
    px = xfl.prices(t, start="1990-01-01", end=END, fields=["adj_close"]).sort_values("date")
    s = px.set_index("date")["adj_close"].astype(float)
    starts[t] = s.index[0]
    ma = s.rolling(200).mean()
    below = (s < ma) & ma.notna()
    cross = below & ~below.shift(1, fill_value=False).astype(bool)

    for peak_i, trough_i, peak_v, trough_v, recovered in drawdown_episodes(s):
        if peak_i < MIN_HISTORY:
            continue  # the 200-day average does not exist yet
        seg = s.values[peak_i:trough_i + 1]
        r20_i = peak_i + int(np.argmax(seg <= peak_v * 0.8))
        ma_hits = np.where(cross.values[peak_i + 1:trough_i + 1])[0]
        ma_i = peak_i + 1 + int(ma_hits[0]) if len(ma_hits) else None
        rec = {"ticker": t, "peak": s.index[peak_i], "trough": s.index[trough_i],
               "decline": trough_v / peak_v - 1, "recovered": recovered,
               "trough_days": trough_i - peak_i,
               "ma_date": s.index[ma_i] if ma_i is not None else pd.NaT}
        for tag, i in (("r20", r20_i), ("ma", ma_i)):
            rec[tag + "_days"] = np.nan if i is None else i - peak_i
            rec[tag + "_done"] = np.nan if i is None else (s.values[i] / peak_v - 1) / (trough_v / peak_v - 1)
            rec[tag + "_further"] = np.nan if i is None else trough_v / s.values[i] - 1
        rows.append(rec)

    # every downside crossing of the 200-day average, once the average exists
    for i in np.where(cross.values)[0]:
        if i < MIN_HISTORY:
            continue
        fwd = s.values[i:i + 127]
        crossings.append({"ticker": t, "date": s.index[i], "worst_fwd": fwd.min() / s.values[i] - 1,
                          "full_window": len(fwd) == 127})

ep = pd.DataFrame(rows).sort_values(["peak", "ticker"]).reset_index(drop=True)
cr = pd.DataFrame(crossings)
openers = set(zip(ep["ticker"], ep["ma_date"]))
cr["opened_a_decline"] = [(t, d) in openers for t, d in zip(cr["ticker"], cr["date"])]

print("=== How late does a bear-market signal arrive? ===")
print(f"Six index funds, split-adjusted closes to {END} (price returns, distributions excluded)")
print("  " + "  ".join(f"{t} from {starts[t]:%Y-%m-%d}" for t in TICKERS))
print(f"{len(ep)} declines of 20% or more from a running peak, "
      f"{int(ep['recovered'].sum())} of them since recovered\n")
print(f"{'':6} {'peak':<11}{'trough':<11}{'decline':>8}   "
      f"{'--- 20% label ---':^26}  {'--- 200-day average ---':^26}")
print(f"{'fund':6} {'':<11}{'':<11}{'':>8}   "
      f"{'days':>5}{'done':>7}{'left to fall':>14}  {'days':>5}{'done':>7}{'left to fall':>14}")
for _, r in ep.iterrows():
    print(f"{r['ticker']:6} {r['peak']:%Y-%m-%d} {r['trough']:%Y-%m-%d} {r['decline']:>7.1%}   "
          f"{r['r20_days']:>5.0f}{r['r20_done']:>7.0%}{r['r20_further']:>14.1%}  "
          f"{r['ma_days']:>5.0f}{r['ma_done']:>7.0%}{r['ma_further']:>14.1%}")

print(f"\nMedians across the {len(ep)} declines")
print(f"  peak-to-trough decline {ep['decline'].median():.1%}, "
      f"trough reached {ep['trough_days'].median():.0f} trading days after the peak")
for tag, name in (("r20", "20% label     "), ("ma", "200-day cross ")):
    print(f"  {name} fires {ep[tag + '_days'].median():>5.0f} days after the peak   "
          f"{ep[tag + '_done'].median():>4.0%} of the decline already done   "
          f"{ep[tag + '_further'].median():>6.1%} still to fall")

full = cr[cr["full_window"]]
print(f"\n200-day average crossed from above {len(cr)} times across the six funds")
print(f"  {cr['opened_a_decline'].sum()} of those crossings opened one of the declines above "
      f"({cr['opened_a_decline'].mean():.1%})")
print(f"  worst close over the next six months, median {full['worst_fwd'].median():.1%}, "
      f"{(full['worst_fwd'] <= -0.10).mean():.1%} fell a further 10% or more")

fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0a0a0a")
ax.set_facecolor("#0a0a0a")
for tag, colour, label in (("ma", "#3b82f6", "First close below the 200-day average"),
                           ("r20", "#f59e0b", "First close 20% below the peak")):
    ax.scatter(ep[tag + "_days"], ep[tag + "_done"] * 100, s=70, alpha=0.85,
               color=colour, edgecolors="#0a0a0a", label=label)
    ax.axhline(ep[tag + "_done"].median() * 100, color=colour, ls="--", lw=1, alpha=0.6)
ax.set_xscale("log")
ax.set_xticks([5, 10, 25, 50, 100, 200, 400])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Trading days from the peak to the signal (log scale)", color="#e0e0e0")
ax.set_ylabel("Share of the eventual decline already realised (%)", color="#e0e0e0")
ax.set_title(f"How late a bear-market signal arrives: {len(ep)} declines in six index funds",
             color="#e0e0e0", fontsize=13)
ax.tick_params(colors="#e0e0e0")
for spine in ax.spines.values():
    spine.set_color("#333333")
leg = ax.legend(facecolor="#0a0a0a", edgecolor="#333333", labelcolor="#e0e0e0", loc="upper left")
leg.get_frame().set_alpha(1)
plt.tight_layout()
plt.savefig("how-late-is-a-bear-market-signal-python.png", dpi=150, facecolor="#0a0a0a")

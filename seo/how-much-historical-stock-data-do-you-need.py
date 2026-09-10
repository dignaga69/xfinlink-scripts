# Full write-up: https://xfinlink.com/blog/how-much-historical-stock-data-do-you-need
import xfinlink as xfl
import pandas as pd

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

# 1. How deep does the price history run for one name?
px = xfl.prices("KO", start="1996-01-01", end="2025-12-31", fields=["close"])
print(f"KO: {len(px):,} sessions, {px['date'].min():%Y-%m-%d} to {px['date'].max():%Y-%m-%d}")

# 2. How many companies left the S&P 500 inside each lookback window?
pages, offset = [], 0
while True:
    page = xfl.index_events(
        "sp500", event_type="removed",
        start="1996-01-01", end="2025-12-31", offset=offset,
    )
    pages.append(page)
    offset += len(page)
    if len(page) < 1000:
        break

ev = pd.concat(pages, ignore_index=True)
ev["year"] = pd.to_datetime(ev["effective_date"]).dt.year

for years in (5, 10, 20, 30):
    n = ev[ev["year"] > 2025 - years].shape[0]
    print(f"last {years:>2} years: {n:>3} removals")

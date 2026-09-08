# Full write-up: https://xfinlink.com/blog/bulk-stock-data-download-vs-api
#
# One year of daily closes for ten large caps: what an API returns when the
# request is filtered before transfer, rather than downloaded as whole files.

import time

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["AAPL", "MSFT", "JNJ", "XOM", "JPM", "PG", "CAT", "KO", "UNH", "HON"]

t0 = time.time()
df = xfl.prices(tickers, start="2025-09-08", end="2026-09-04", fields=["close"])
elapsed = time.time() - t0

print(f"rows: {len(df)}   tickers: {df['ticker'].nunique()}   seconds: {elapsed:.1f}")
print(f"memory: {df.memory_usage(deep=True).sum() / 1024:.0f} KB")
print(f"sessions per ticker: {df.groupby('ticker').size().unique()}")

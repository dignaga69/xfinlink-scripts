# Full write-up: https://xfinlink.com/blog/list-of-all-us-stock-tickers
#
# Pages the entire US entity universe out of xfl.search() and describes what
# came back: how many entities, how many carry a ticker, and how many ticker
# strings ended their life on more than one company.

import pandas as pd
import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

PAGE = 500  # per-call maximum for search
frames, offset, calls = [], 0, 0
while True:
    page = xfl.search(limit=PAGE, offset=offset)
    calls += 1
    frames.append(page)
    if len(page) < PAGE:
        break
    offset += PAGE

universe = pd.concat(frames, ignore_index=True)

print(f"entities         {len(universe):,}")
print(f"HTTP calls       {calls}")
print(f"unique entity_id {universe['entity_id'].nunique():,}")
print()
print(universe["entity_type"].value_counts().to_string())
print()
print(f"rows with a ticker  {universe['ticker'].notna().sum():,}")
print(f"distinct tickers    {universe['ticker'].nunique():,}")
print(f"with a GICS sector  {universe['gics_sector'].notna().sum():,}")
print(f"with an SIC code    {universe['sic'].notna().sum():,}")

names_per_ticker = universe.groupby("ticker")["entity_name"].nunique()
print(f"tickers ending on two or more companies  {(names_per_ticker >= 2).sum():,}")

print()
print("OPEN:")
for e in xfl.resolve("OPEN")["data"]["OPEN"]["entities"]:
    print(f"  {e['entity_id']:>6}  {e['name']:<28}  {e['ticker_valid_from']} to {e['ticker_valid_to']}")

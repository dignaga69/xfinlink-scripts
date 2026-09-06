# Full write-up: https://xfinlink.com/blog/how-to-get-stock-data-into-excel
# Builds a watchlist report and writes it to watchlist.xlsx and watchlist.csv.
# .to_excel() needs openpyxl: pip install openpyxl

import xfinlink as xfl

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

WATCH = ["AAPL", "MSFT", "JPM", "UNH", "CVX", "HD"]

px = xfl.prices(WATCH, period="1w", fields=["close"])
last_price = px.sort_values("date").groupby("entity_id", as_index=False).tail(1)

mt = xfl.metrics(WATCH, period_type="ttm", period="1y",
                 fields=["pe_ratio", "net_margin"])
last_metric = mt.sort_values("period_end").groupby("entity_id", as_index=False).tail(1)

report = last_price.merge(last_metric, on=["entity_id", "ticker", "entity_name"])
report = report[["entity_id", "ticker", "entity_name", "date", "close",
                 "period_end", "pe_ratio", "net_margin"]]

report.to_excel("watchlist.xlsx", index=False)
report.to_csv("watchlist.csv", index=False)
print(report.to_string(index=False))

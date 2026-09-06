# How to Get Stock Data Into Excel With Python

Pull the numbers with a short Python script, write them to a CSV or an .xlsx file, and point Excel at that file with a query it can refresh. Around twenty lines of code cover the whole path, and the same file feeds Google Sheets, Power BI or anything else that reads a CSV without anything upstream changing. Why bother, when a spreadsheet can fetch quotes from a formula? Because the built-in functions return price history and nothing else, and the one in Google Sheets cannot be reached from a script at all, which rules it out for any report that has to rebuild itself on a schedule.

## What Do the Built-In Spreadsheet Functions Return?

Excel ships `STOCKHISTORY`. Microsoft's documentation, read on 6 September 2026, says the function "requires a Microsoft 365 Personal, Microsoft 365 Family, Microsoft 365 Business Standard, or Microsoft 365 Business Premium subscription", and lists six properties it can return: date, close, open, high, low and volume, at daily, weekly or monthly intervals. The same page notes that it "generally only updates after a trading day completes". For a personal sheet tracking closing prices, that is the shortest route available, and it costs nothing beyond a subscription many desks already hold.

Google Sheets has `GOOGLEFINANCE`, whose historical mode is narrower. Google's documentation on 6 September 2026 states that once a date parameter appears "the request is considered historical and only the historical attributes are allowed" (open, close, high, low, volume and `all`), and, further down the same page, that "Historical data cannot be downloaded or accessed via the Sheets API or Apps Script." One line on that page is worth reading before a sheet goes to a client: "The data is not for financial industry professional use or use by other professionals at non-financial firms (including government entities)."

Neither function returns revenue, margins, share counts, index membership, insider transactions or institutional holdings. A quarterly review pack built on either one gets its fundamentals somewhere else, usually by hand, and hand-entered cells are the part of a recurring report that quietly goes stale.

## How Do You Pull the Data in Python?

Six names, one price call, one metrics call, and a file on disk:

```python
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
```

```
 entity_id ticker            entity_name       date  close period_end  pe_ratio  net_margin
      3616     HD         HOME DEPOT INC 2026-09-04 321.05 2026-08-02     22.47    0.084096
      1537    JPM    JPMORGAN CHASE & CO 2026-09-04 358.64 2026-06-30     15.36    0.326301
      1553    CVX           CHEVRON CORP 2026-09-04 208.60 2026-06-30     20.06    0.098658
         1   AAPL              Apple Inc 2026-09-04 319.97 2026-06-27     36.74    0.276186
      8611   MSFT         MICROSOFT CORP 2026-09-04 499.70 2026-06-30     27.84    0.403054
      7688    UNH UNITEDHEALTH GROUP INC 2026-09-04 397.14 2026-06-30     25.49    0.031373
```

Two details make this shorter than it looks. `fields=` limits the payload to the columns the sheet displays, which keeps the row count small. Trailing-twelve-month ratios come back computed, so no cell in the workbook has to divide one number by another and no analyst has to remember which quarter the denominator came from. Writing `.xlsx` needs openpyxl installed alongside pandas; the pandas documentation lists it for "Reading / writing for Excel 2010 xlsx/xlsm/xltx/xltm files". `pip install openpyxl` and the line works.

Keep a `period_end` column in the sheet. A report showing a price from Friday and a margin from a filing four months old is correct, and it is only obviously correct if both dates sit in the row.

## How Do You Keep the Report Pointed at the Same Companies?

Symbols move between companies. A watchlist keyed on the letters "HD" is a watchlist that will one day describe whoever holds those letters, and the sheet will not announce the swap: the column header stays the same and the numbers change underneath it.

Every row xfinlink returns carries an `entity_id` alongside the ticker, so the fix is to save the identifiers on the first run and pull by identifier afterwards:

```python
px = xfl.prices(WATCH, period="1w", fields=["close"])
ids = px.groupby("ticker")["entity_id"].first().to_dict()
# {'AAPL': 1, 'CVX': 1553, 'HD': 3616, 'JPM': 1537, 'MSFT': 8611, 'UNH': 7688}

df = xfl.prices(entity_id=list(ids.values()), period="1w", fields=["close"])
```

The identifier survives a rename, a symbol change and a symbol reassignment. `xfl.resolve("HD")` returns two companies for that one symbol, only one of which trades under it today, which is the case the ticker-keyed sheet gets wrong. For a report covering an index rather than a hand-picked list, `xfl.index("sp500", as_of="2020-06-30")` returns the roster as it stood on that date instead of today's roster, so a historical tab does not silently fill with companies that joined later.

## How Does the Workbook Pick the File Up?

Excel imports a local file through Power Query: **Data > Get Data > From File > From Text/CSV**, per Microsoft's documentation on 6 September 2026. Once the query exists, the connection can be refreshed on demand with **Data > Refresh All**, and the same documentation describes two unattended options in Connection Properties: a "Refresh data when opening the file" checkbox, and a "Refresh every" checkbox taking a number of minutes.

That leaves the script itself, which cron on Linux or macOS and Task Scheduler on Windows will run at a fixed time. One habit is worth adopting on day one: write to a temporary name and rename it into place, so a refresh that fires mid-write reads the previous file rather than half of the new one.

```python
import os

report.to_csv("watchlist.tmp.csv", index=False)
os.replace("watchlist.tmp.csv", "watchlist.csv")
```

For Google Sheets, the same file arrives through **File > Import**, which Google's documentation on 6 September 2026 lists as accepting ".csv, .txt, .tsv, .tab", or the script writes cells directly through the Sheets API, which documents `spreadsheets.values.update`, `batchUpdate` and `append` for writing values. Since GOOGLEFINANCE history is unavailable to Apps Script by Google's own documentation, a pushed file or an API write is the only Sheets route that runs without a person opening the tab. Power BI reads the same file through the Text/CSV option in **Get Data**, a connector Microsoft's Power Query documentation lists for both Excel and Power BI, so one scheduled script serves the workbook, the sheet and the dashboard.

## What Does a Daily Refresh Cost?

One request per ticker per endpoint. The six-name report above spends 12 requests a day, inside the free tier's 100 a day (capped at 40 in any hour); a free key carries one ticker per call, so the same job runs as twelve single-name calls and the request count comes out the same. A sixty-name watchlist on the same two endpoints spends 120 a day and needs a paid plan. xfinlink Pro is $29 a month with 10,000 requests a day and carries up to 100 tickers in a single call, which is the version of the script shown above.

| Route | What it can return | Where the pull runs | Free ceiling (September 2026) |
|---|---|---|---|
| Excel `STOCKHISTORY` | date, close, open, high, low, volume; daily, weekly or monthly | Inside the workbook | Requires a Microsoft 365 Personal, Family, Business Standard or Business Premium subscription |
| Sheets `GOOGLEFINANCE` | open, close, high, low, volume over a date range | Inside the sheet; "cannot be downloaded or accessed via the Sheets API or Apps Script" | Free with Sheets; "not for financial industry professional use" |
| Alpha Vantage add-ons | API data pulled into Excel or Sheets cells | Inside the workbook or sheet | "25 API requests per day"; `TIME_SERIES_DAILY` takes one `symbol` |
| Python script writing a file | Prices, fundamentals, computed metrics, index membership, insider transactions, 13F holdings | Wherever the script is scheduled | 100 requests/day, max 40 per hour, rolling 1-year history; insider and 13F endpoints on paid plans |

Alpha Vantage publishes official add-ons for Office 365 Excel and Google Sheets (alphavantage.co/spreadsheets, 6 September 2026), which is less setup than a scheduled script for anyone who wants formulas in cells. The ceiling arrives quickly: its support page states a free key allows "25 API requests per day", and `TIME_SERIES_DAILY` takes a single `symbol` per call, so a thirty-name daily refresh runs past the free allowance before it finishes. Multi-symbol calls exist on the Realtime Bulk Quotes endpoint, which its documentation marks premium and which serves quotes rather than history.

yfinance is the other common source for a script like this, and for a personal sheet it works. Its package page states that it is "not affiliated, endorsed, or vetted by Yahoo, Inc.", that it is "intended for research and educational purposes", and that "the Yahoo! finance API is intended for personal use only" — which is the line that matters once the workbook leaves your own machine.

## FAQ

**Does Excel have to be open for the refresh to work?**

No. The script writes the file on its own schedule; the workbook reads whatever is there when it refreshes, either on open or on the timed interval set in Connection Properties.

**Can the sheet hold fundamentals as well as prices?**

Yes, and that is the main reason to move the pull into Python. One `xfl.fundamentals()` or `xfl.metrics()` call returns statement lines and computed ratios in the same shape as the price frame, so both land in the same file.

**What happens when a company in the watchlist changes its ticker?**

Nothing, if the report pulls by `entity_id`. The row keeps its history and the new symbol appears in the `ticker` column. A sheet keyed on the old symbol either returns an error or, worse, returns a different company's numbers under the old heading.

Full endpoint reference is in the [API documentation](https://xfinlink.com/docs) and plan limits are on the [pricing page](https://xfinlink.com/pricing). For the identifier question in more depth, see [how to track companies through ticker changes](https://xfinlink.com/blog/track-companies-ticker-changes-python); for swapping an existing script over, [how to replace yfinance in a Python script](https://xfinlink.com/blog/how-to-replace-yfinance); and for sizing a bigger job, [financial data API rate limits](https://xfinlink.com/blog/financial-data-api-rate-limits).

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*

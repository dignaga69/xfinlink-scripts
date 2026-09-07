# How to Get a List of All US Stock Tickers

Two free files answer the narrow version of this question. The Nasdaq symbol directory (`nasdaqlisted.txt` plus `otherlisted.txt`) lists every symbol trading on a US exchange right now, and the SEC's `company_tickers.json` maps ticker to CIK for every company that files with the SEC. Both describe today. If the list has to include companies that stopped trading, or has to join to prices and financial statements afterwards, neither file will do it, and the job becomes a query against a data API rather than a download.

## Which Ticker List Do You Actually Want?

The phrase "all US stock tickers" covers three different requests that need three different sources.

A list of **what trades today** is a listings question, and the exchanges publish it. A list of **everything that has ever traded** is a history question, and no live directory holds it, because a symbol leaves the file on the day the company leaves the market. A list of **the names worth analysing** is a universe question, usually answered better by an index roster than by a dump of every symbol.

Most people who search for the first are really after the second or the third. A screener built on today's symbols and run over ten years of history quietly deletes every company that failed, merged or went private during the window, which is the standard route into [survivorship bias](https://xfinlink.com/blog/what-is-survivorship-bias-in-backtesting).

## Where Do the Free Symbol Files Come From?

The Nasdaq symbol directory at `nasdaqtrader.com/dynamic/symdir/` publishes two pipe-delimited files: `nasdaqlisted.txt` for Nasdaq issues and `otherlisted.txt` for everything on the other US exchanges. The pair carried 13,155 rows flagged as non-test issues in the file stamped 4 September 2026, with an ETF column separating funds from operating companies. No history, no company identifier beyond the symbol, and no sector.

The SEC's `company_tickers.json` is the other free option, and it answers a different question than people assume. Fetched on 7 September 2026 it held about 10,400 records across roughly 8,000 distinct CIK numbers, since a company with two share classes appears twice. Two pulls a little over an hour apart that morning differed by three records, so the file is refreshed continuously and any exact count is good for a moment rather than a day. It covers SEC filers rather than listed securities, so it includes companies whose shares do not trade on an exchange and excludes nothing on the way out. There are no validity dates in the file.

Alpha Vantage is worth naming here because its `LISTING_STATUS` endpoint does something the two files above do not. Its documentation, read on 7 September 2026, states that it "returns a list of active or delisted US stocks and ETFs, either as of the latest trading day or at a specific time in history" and that "any YYYY-MM-DD date later than 2010-01-01 is supported". A call to the documented URL returned 14,409 active rows the same day, with `ipoDate` and `delistingDate` columns attached. For the question "what was listed on 3 August 2013", that endpoint is a genuinely good free answer, and its 2010 floor is the thing to check against the period a study needs.

| Source | What it returns | History | Identifier | Joins to prices and fundamentals |
|---|---|---|---|---|
| Nasdaq symbol directory | Currently listed symbols (13,155 non-test rows, 4 Sep 2026 file) | None | Symbol only | No |
| SEC `company_tickers.json` | Current SEC filers with a ticker (about 10,400 records, 8,000 CIKs) | None | Ticker, CIK, name | Through further EDGAR calls |
| Alpha Vantage `LISTING_STATUS` | Active or delisted US stocks and ETFs (14,409 active rows, 7 Sep 2026) | Any date after 2010-01-01, per its documentation | Symbol | Separate calls, keyed on symbol |
| `xfl.search()` | Every US entity on file, listed or not | Full registry, no date floor | `entity_id`, ticker, SIC, NAICS, GICS | Yes, by `entity_id` |

## How Do You Pull the Whole US Universe in Python?

`xfl.search()` returns entities rather than symbols, and with no filters it returns all of them. The per-call maximum is 500 rows, so a full pass is a loop over `offset`.

```python
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
```

```
entities         46,502
HTTP calls       94
unique entity_id 46,502

entity_type
corporation        40924
etf                 5575
closed_end_fund        3

rows with a ticker  45,665
distinct tickers    42,467
with a GICS sector  38,255
with an SIC code    41,404
tickers ending on two or more companies  2,731
```

Ninety-four calls, no duplicate ids, and 46,502 entities against the 13,155 symbols in the current listings file. The gap is the part of the market that has already been and gone: New York Central Railroad, whose hold on the ticker CN runs from 1925 to 1968 in the registry, sits in that file alongside Apple. Each row carries the classification codes as well as the name, so filtering the universe to one sector or one SIC code happens in pandas rather than in a second round of lookups.

The request count is worth planning for. A free key allows 100 requests a day with no more than 40 in an hour, so a full universe pass spans a few hours on the free tier and runs in one go on any paid plan; the [pricing page](https://xfinlink.com/pricing) has the per-tier figures. Most callers do not need the full pass more than once, since `xfl.search(gics_sector="Energy")` or `xfl.search(sic="3711")` narrows the request at the source.

## Why Is the Ticker Not the Identifier?

In the pull above, 45,665 rows carry a ticker but only 42,467 ticker strings are distinct. On 2,731 of those strings, two or more differently named companies ended their run. Symbols get released and reassigned. A list keyed on the string alone therefore silently merges companies that have nothing to do with each other.

```
OPEN:
   18214  OPEN ENVIRONMENT CORP         1995-04-13 to 1996-11-18
   14050  OPENROUTE NETWORKS INC        1998-06-17 to 1999-12-22
   28927  OPEN SOLUTIONS INC            2003-11-26 to 2007-01-23
   33554  OPENTABLE INC                 2009-05-21 to 2014-07-23
   19789  OPENDOOR TECHNOLOGIES INC     2020-12-21 to None
```

Five companies, one symbol, three decades. `xfl.resolve("OPEN")` returns them with the dates each held the ticker, and the `entity_id` on the left is what the price, fundamentals, metrics, insider and holdings functions accept in place of a symbol. Passing `entity_id=33554` reaches OpenTable's own history, which the string "OPEN" no longer points at. The wider identifier question is covered in [ticker vs CIK vs FIGI](https://xfinlink.com/blog/ticker-vs-cik-vs-figi), and the delisted case in [historical data for delisted stocks](https://xfinlink.com/blog/how-to-get-delisted-stock-data).

## When Is an Index Roster the Better Universe?

Very few analyses want 46,502 names. Screens, factor tests and comparable-company work all want a defined universe with a membership rule, and for those the index endpoint replaces the whole exercise: `xfl.index("sp500")` returned 504 rows today and `xfl.index("ndx100")` returned 101, one call each, with `russell2000` and `djia` available on the same function. Adding `as_of="2014-06-30"` returns the roster as it stood on that date rather than today's, which is the difference between a backtest that tests a strategy and one that tests hindsight.

Start with the index if the work has a benchmark. Fall back to the full entity pull when the question is genuinely market-wide, such as counting how many companies in a sector have ever filed, or building a mapping table that has to survive a renaming.

## FAQ

**How many US stock tickers are there?**
About 13,000 symbols trade on US exchanges at any moment, which was 13,155 non-test rows in the 4 September 2026 Nasdaq directory file. Counting every company that has traded raises the figure several times over; the xfinlink entity registry held 46,502 US entities on 7 September 2026, of which 40,924 are corporations and 5,575 are ETFs.

**Can I get a list of delisted tickers?**
Yes, from a source that keeps history. Alpha Vantage's `LISTING_STATUS` endpoint serves a delisted list for any date after 2010-01-01 according to its documentation. `xfl.search()` returns delisted and live entities together with no date floor, and each carries an `entity_id` that still reaches its price and statement history.

**Does the symbol list need an API key?**
The Nasdaq and SEC files need none. `xfl.search()` answers without a key under a per-IP limit of 60 requests an hour, and a key moves the request onto the account's own allowance instead; the [docs](https://xfinlink.com/docs) list the limits per tier.

**Why does the same ticker appear against several companies?**
Because exchanges reuse symbols after a company leaves. Any list that treats the string as a primary key will collide on 2,731 of them in the US registry, which is why entity ids exist.

*Built with [xfinlink](https://xfinlink.com) — free financial data API for Python. `pip install -U xfinlink`*

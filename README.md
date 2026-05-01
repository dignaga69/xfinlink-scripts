# xfinlink

Free financial data API for US equities. Six endpoints, 30+ years of history, built-in entity resolution for ticker recycling.

## Install

```bash
pip install xfinlink
```

## Quick start

```python
import xfinlink as xfl

xfl.set_api_key("your_key")  # free at https://xfinlink.com/signup

df = xfl.prices("AAPL", period="1y")
print(df[["date", "close", "volume"]].head())
```

## Endpoints

| Endpoint | Function | Returns | API key required |
|----------|----------|---------|-----------------|
| `/v1/prices` | `xfl.prices()` | DataFrame | Yes |
| `/v1/fundamentals` | `xfl.fundamentals()` | DataFrame | Yes |
| `/v1/metrics` | `xfl.metrics()` | DataFrame | Yes |
| `/v1/resolve` | `xfl.resolve()` | dict | No |
| `/v1/search` | `xfl.search()` | DataFrame | No |
| `/v1/index` | `xfl.index()` | DataFrame | No |

---

## Prices

Daily OHLCV data. Split-adjusted by default.

```python
# Single ticker, last month
df = xfl.prices("AAPL", period="1mo", limit=5)
```

| entity_id | ticker | entity_name | date | open | high | low | close | volume |
|-----------|--------|-------------|------|------|------|-----|-------|--------|
| 1 | AAPL | Apple Inc | 2026-04-01 | 254.08 | 256.18 | 253.33 | 255.63 | 40059400 |
| 1 | AAPL | Apple Inc | 2026-04-02 | 254.20 | 256.13 | 250.65 | 255.92 | 31289400 |
| 1 | AAPL | Apple Inc | 2026-04-06 | 256.51 | 262.16 | 256.46 | 258.86 | 29329900 |

Full column list: `entity_id`, `ticker`, `entity_name`, `gics_sector`, `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `return_daily`, `shares_outstanding`, `exchange_code`, `split_ratio`, `dividend`. `market_cap` is available via `fields=["market_cap"]` but not included in the default response.

```python
# Multiple tickers
df = xfl.prices(["AAPL", "MSFT"], period="1w")

# Specific date range
df = xfl.prices("AAPL", start="2024-01-02", end="2024-01-05")

# Select specific fields (structural fields always included)
df = xfl.prices("AAPL", period="1w", fields=["close", "volume"])
# → columns: entity_id, ticker, entity_name, gics_sector, date, close, volume
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker` | str or list | Ticker(s). `"AAPL"`, `["AAPL", "MSFT"]`, `"AAPL,MSFT"`. Max 50. |
| `start` | str or date | Start date, `"YYYY-MM-DD"`. Default: 1 year ago. |
| `end` | str or date | End date, `"YYYY-MM-DD"`. Default: today. |
| `period` | str | Shorthand: `"1w"`, `"1mo"`, `"3mo"`, `"6mo"`, `"1y"`, `"5y"`, `"10y"`, `"ytd"`, `"max"`. |
| `fields` | list or str | Fields to return beyond structural defaults. |
| `adjust` | str | `"split"` (default) or `"none"`. |
| `limit` | int | Max rows returned. |

---

## Fundamentals

Financial statements: income statement, balance sheet, cash flow. 147 data fields including industry-specific columns for banks, insurance, and REITs.

```python
# Annual fundamentals
df = xfl.fundamentals("AAPL", period_type="annual", period="3y", limit=3)
```

| entity_name | period_end | period_type | revenue | net_income | total_assets | total_equity |
|-------------|-----------|-------------|---------|------------|-------------|-------------|
| Apple Inc | 2021-09-30 | annual | 365817 | 94680 | 351002 | 63090 |
| Apple Inc | 2022-09-30 | annual | 394328 | 99803 | 352755 | 50672 |
| Apple Inc | 2023-09-30 | annual | 383285 | 96995 | 352583 | 62146 |

```python
# Quarterly
df = xfl.fundamentals("AAPL", period_type="quarterly", period="1y", limit=4)

# Select specific fields
df = xfl.fundamentals("AAPL", period_type="annual", fields=["revenue", "net_income", "eps_diluted"], period="3y")
# → columns: entity_id, ticker, entity_name, gics_sector, period_end,
#            period_type, fiscal_year, fiscal_period, filing_date, version,
#            source, revenue, net_income, eps_diluted
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker` | str or list | Ticker(s). |
| `period_type` | str | `"annual"`, `"quarterly"`, or `"all"` (default). |
| `version` | str | `"restated"` (default), `"original"`, or `"all"`. |
| `fields` | list or str | Specific fields, e.g. `["revenue", "net_income"]`. |
| `period_only` | bool | De-cumulate YTD cash flows to single-quarter values. |
| `start`, `end`, `period`, `limit` | | Same as prices. |

### Industry-specific fields

Banks, insurance companies, and REITs have additional columns populated alongside standard fields.

```python
df = xfl.fundamentals("JPM", period_type="annual", period="3y", limit=1)
```

**Bank fields** (JPMorgan, 2024):

| Field | Value |
|-------|-------|
| `bank_interest_income` | 193,933 |
| `bank_interest_expense` | 101,350 |
| `bank_net_interest_income` | 81,905 |
| `bank_noninterest_income` | 84,973 |
| `bank_noninterest_expense` | 91,797 |
| `bank_loans_net` | 1,323,643 |
| `bank_deposits` | 2,406,032 |
| `bank_allowance_for_credit_losses` | 24,345 |
| `bank_net_charge_offs` | 8,638 |

Also available: `ins_*` fields for insurance and `reit_*` fields for REITs.

---

## Metrics

Computed ratios and per-share values. 37 computed metrics derived from prices + fundamentals.

```python
df = xfl.metrics("AAPL", period_type="annual", period="3y", limit=3)
```

| entity_name | period_end | pe_ratio | roe | gross_margin | net_margin |
|-------------|-----------|----------|-----|-------------|-----------|
| Apple Inc | 2023-09-30 | 27.93 | 1.56 | 0.463 | 0.253 |
| Apple Inc | 2024-09-28 | 37.47 | 1.65 | 0.462 | 0.240 |
| Apple Inc | 2025-09-27 | 36.37 | 1.52 | 0.469 | 0.269 |

```python
# Select specific metrics
df = xfl.metrics("AAPL", fields=["pe_ratio", "roe", "gross_margin"], period="3y")
```

All 37 metric columns: `market_cap`, `market_cap_diluted`, `enterprise_value`, `pe_ratio`, `ps_ratio`, `pb_ratio`, `price_to_cash_flow`, `ev_ebitda`, `earnings_yield`, `dividend_yield`, `gross_margin`, `operating_margin`, `net_margin`, `ebitda_margin`, `roe`, `roa`, `roic`, `debt_to_equity`, `debt_to_assets`, `long_term_debt_to_equity`, `long_term_debt_to_assets`, `current_ratio`, `quick_ratio`, `interest_coverage`, `asset_turnover`, `inventory_turnover`, `revenue_per_share`, `book_value_per_share`, `tangible_book_value_per_share`, `cash_per_share`, `debt_per_share`, `ocf_per_share`, `fcf_per_share`, `ebit_per_share`, `ebitda_per_share`, `capex_per_share`, `working_capital_per_share`.

---

## Resolve

Entity resolution and history for a ticker. This is what makes xfinlink different from other financial APIs: it correctly handles ticker recycling, name changes, and corporate events.

```python
info = xfl.resolve("DELL")  # → dict
```

DELL resolves to **two entities** because Dell Inc went private in 2013 and Dell Technologies re-listed in 2018:

```python
entities = info["data"]["DELL"]["entities"]

# Entity 1: Dell Inc (1988-2013)
# {
#   "entity_id": 10408,
#   "name": "DELL INC",
#   "ticker_valid_from": "1988-06-22",
#   "ticker_valid_to": "2013-10-29",
#   "index_membership": [{"index": "SP500", "added": "1996-09-06", "removed": "2013-10-28"}]
# }

# Entity 2: Dell Technologies Inc (2018-present)
# {
#   "entity_id": 65047,
#   "name": "DELL TECHNOLOGIES INC",
#   "ticker_valid_from": "2018-12-28",
#   "ticker_valid_to": null,
#   "index_membership": [{"index": "SP500", "added": "2024-09-23", "removed": null}]
# }
```

### GM: pre/post bankruptcy

```python
info = xfl.resolve("GM")
```

GM resolves to two distinct entities:

| Entity | Valid dates | Notes |
|--------|------------|-------|
| General Motors Corporation (pre-2009 bankruptcy) | 1962-07-02 to 2009-06-01 | Old GM, liquidated |
| General Motors Company | 2010-11-18 to present | New GM, post-IPO |

### FB: ticker recycling across companies

```python
info = xfl.resolve("FB")
```

"FB" was used by **four different companies** over 50 years:

| Entity | Valid dates | Company |
|--------|------------|---------|
| BankBoston Corp | 1971 - 1983 | Banking |
| Falcon Building Products Inc | 1994 - 1997 | Manufacturing |
| FBR Asset Investment Corp | 1999 - 2003 | Finance |
| Meta Platforms Inc | 2012 - 2022 | Social media (renamed from Facebook) |

### META: checking the current ticker

```python
info = xfl.resolve("META")
# Meta Platforms Inc, valid from 2022-06-09 (when FB ticker changed to META)
```

### Batch resolve

```python
info = xfl.resolve(["AAPL", "MSFT", "INVALID"])
# info["meta"]["tickers_resolved"]   → ["AAPL", "MSFT"]
# info["meta"]["tickers_unresolved"] → ["INVALID"]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ticker` | str or list | Ticker(s) to resolve. |
| `include` | list | Sections to include: `["index", "classifications"]`. Default: all. |

Each resolved entity includes: `entity_id`, `name`, `entity_type`, `country`, `ticker_valid_from`, `ticker_valid_to`, `classifications` (SIC, NAICS, GICS), `index_membership`.

---

## Search

Find entities by name, sector, type, or industry codes.

```python
df = xfl.search(q="apple")
```

| entity_id | ticker | entity_name | entity_type | gics_sector |
|-----------|--------|-------------|-------------|-------------|
| 10246 | 7076B | APPLE BANCORP INC | corporation | Financials |
| 13996 | APLE | APPLE HOSPITALITY REIT INC | corporation | Real Estate |
| 1 | AAPL | Apple Inc | corporation | Information Technology |

```python
# Filter by GICS sector
df = xfl.search(gics_sector="Energy", limit=5)

# Filter by entity type
df = xfl.search(entity_type="etf", q="bond", limit=10)

# Filter by SIC code
df = xfl.search(sic="6020", limit=5)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | str | Fuzzy search on name or ticker. |
| `gics_sector` | str | GICS sector filter, e.g. `"Information Technology"`. |
| `entity_type` | str | `"corporation"`, `"etf"`, `"closed_end_fund"`. |
| `sic` | str | SIC code filter. |
| `naics` | str | NAICS code filter. |
| `limit` | int | Max results (default 50, max 500). |
| `offset` | int | Pagination offset. |

Returns: DataFrame with `entity_id`, `ticker`, `entity_name`, `entity_type`, `gics_sector`, `gics_sub_industry`, `sic`, `naics`, `country`.

---

## Index

S&P 500 constituents, current or point-in-time historical.

```python
# Current S&P 500
df = xfl.index("sp500", limit=5)
```

| entity_id | ticker | entity_name | added_date | removed_date |
|-----------|--------|-------------|------------|-------------|
| 14113 | AES | A E S CORP | 1998-10-02 | None |
| 550 | APA | A P A CORP | 1997-07-28 | None |
| 11169 | ABBV | ABBVIE INC | 2013-01-02 | None |
| 30471 | ACN | ACCENTURE PLC IRELAND | 2011-07-06 | None |
| 8954 | ADBE | ADOBE INC | 1997-05-06 | None |

```python
# Historical: who was in the S&P 500 on Jan 1, 2010?
df = xfl.index("sp500", as_of="2010-01-01", limit=5)
```

| entity_id | ticker | entity_name | added_date | removed_date |
|-----------|--------|-------------|------------|-------------|
| 14113 | AES | A E S CORP | 1998-10-02 | |
| 670 | GAS | A G L RESOURCES INC | 1989-08-03 | 2011-12-12 |
| 17269 | AKS | A K STEEL HOLDING CORP | 2008-07-01 | 2011-12-16 |

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `index_name` | str | Index name. Currently `"sp500"`. |
| `as_of` | str or date | Point-in-time date. Default: current membership. |
| `limit` | int | Max results (default 500). |
| `offset` | int | Pagination offset. |

---

## Utilities

```python
# Check API health
xfl.status()
# → {"status": "ok", "version": "0.5.8", "tables": [...]}

# Get LLM-friendly context block (for AI agents / tool use)
context = xfl.llm_context()
```

---

## What makes xfinlink different

- **Ticker recycling handled correctly.** "FB" was used by 4 different companies. "GM" maps to two separate entities across bankruptcy. xfinlink tracks which company owned a ticker and when.
- **Delisted and historical company data.** Lehman Brothers, Enron, WorldCom, Bear Stearns -- full price and fundamental history.
- **30+ years of history.** Daily prices from 1996. Annual and quarterly fundamentals from the early 1990s.
- **Entity resolution built in.** Every response includes `entity_id` -- a stable identifier that follows companies through ticker changes, name changes, and exchange transfers.
- **Industry-specific fields.** Bank, insurance, and REIT fields alongside standard financial statements. No separate endpoints.
- **Point-in-time S&P 500.** Historical index membership for survivorship-bias-free backtesting.
- **Raw data only.** All values come from filings and market data. No xfinlink-invented classifications or estimates.

## Links

- **Docs:** [https://xfinlink.com](https://xfinlink.com)
- **Get API key:** [https://xfinlink.com/signup](https://xfinlink.com/signup)
- **LLM reference:** [https://xfinlink.com/llms.txt](https://xfinlink.com/llms.txt)
- **PyPI:** [https://pypi.org/project/xfinlink/](https://pypi.org/project/xfinlink/)

## License

MIT

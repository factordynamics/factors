# factors

[![CI](https://github.com/factordynamics/factors/actions/workflows/ci.yml/badge.svg)](https://github.com/factordynamics/factors/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

factors is a unified factor library for quantitative finance, serving both alpha generation and risk modeling use cases. It provides a common implementation of equity factors including momentum, value, quality, size, volatility, growth, and liquidity. The library eliminates code duplication between alpha models (tarifa) and risk models (perth) by providing a single source of truth for factor definitions.

Factor exposures are computed as cross-sectionally standardized z-scores: z_i = (x_i - mean) / std. This ensures comparability across factors and time periods.

## Scope

factors is a computation library, not a data library. It takes market data (prices, fundamentals) as input and produces factor exposures as output. Data fetching, caching, and storage should be handled upstream. The library answers "what are the factor exposures?" not "where do I get the data?"

## Quick Start

List all available factors:

```bash
just factors
```

Show factor information:

```bash
just info short_term_momentum
```

## Factors

| Category | Factors | Description |
|----------|---------|-------------|
| **Momentum** | short_term, medium_term, long_term | Price trend persistence (1mo, 6mo, 12mo) |
| **Value** | book_to_price, earnings_yield, fcf_yield | Relative valuation metrics |
| **Quality** | roe, roa, profit_margin, leverage | Profitability and financial health |
| **Size** | log_market_cap | Company scale |
| **Volatility** | market_beta, historical_volatility | Risk measures |
| **Growth** | earnings_growth, sales_growth | YoY growth rates |
| **Liquidity** | turnover_ratio, amihud_illiquidity | Trading cost measures |

## Development

Requires Rust 1.88+ and [just](https://github.com/casey/just). Run `just ci` to ensure all tests and lints pass.

## Attribution

Built on [toraniko-rs](https://github.com/factordynamics/toraniko-rs), a Rust port of the original [toraniko](https://github.com/0xfdf/toraniko) Python implementation.

## License

MIT License - see [LICENSE](LICENSE).

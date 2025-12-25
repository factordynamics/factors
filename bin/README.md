# factors CLI

Command-line interface for the `factors` unified factor library.

## Overview

The `factors` CLI provides easy access to the comprehensive factor library for quantitative finance. It enables factor discovery, introspection, and computation directly from the command line.

## Installation

```bash
cargo install factors-bin
```

Or build from source:

```bash
cargo build --release
./target/release/factors --help
```

## Usage

### List All Available Factors

Display all registered factors grouped by category:

```bash
factors list
```

### Show Factor Information

Get detailed information about a specific factor, including its description, required inputs, and parameters:

```bash
factors info short_term_momentum
factors info book_to_price
factors info market_beta
```

### Compute Factors

Compute a specific factor for a given symbol (placeholder implementation):

```bash
factors compute AAPL --factor short_term_momentum
factors compute MSFT --factor earnings_yield
```

## Available Factors

The CLI provides access to all factors in the library, organized by category:

- **Momentum**: Short-term, medium-term, and long-term momentum
- **Value**: Book-to-price, earnings yield, FCF yield
- **Quality**: ROE, ROA, profit margin, leverage
- **Size**: Log market cap
- **Volatility**: Market beta, historical volatility
- **Growth**: Earnings growth, sales growth
- **Liquidity**: Turnover ratio, Amihud illiquidity

## Examples

```bash
# List all factors
factors list

# Get information about ROE
factors info roe

# Compute short-term momentum for AAPL
factors compute AAPL --factor short_term_momentum
```

## License

MIT

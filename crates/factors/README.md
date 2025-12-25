# factors

Unified factor library for quantitative finance, serving both alpha generation (tarifa) and risk modeling (perth) use cases.

## Features

- **Momentum Factors**: Short-term (1mo), medium-term (6mo), and long-term (12mo) momentum
- **Value Factors**: Book-to-price, earnings yield, and free cash flow yield
- **Quality Factors**: ROE, ROA, profit margins, and leverage
- **Size Factors**: Log market capitalization
- **Volatility Factors**: Market beta and historical volatility
- **Growth Factors**: Earnings growth and sales growth
- **Liquidity Factors**: Turnover ratio and Amihud illiquidity

## Architecture

```text
factors/
├── traits.rs           # Core Factor trait definition
├── registry.rs         # Factor discovery and introspection
├── standardize.rs      # Cross-sectional z-scoring utilities
├── momentum/           # Trend persistence factors
├── value/              # Relative valuation factors
├── quality/            # Profitability and leverage factors
├── size/               # Market capitalization factors
├── volatility/         # Risk and beta factors
├── growth/             # Growth rate factors
└── liquidity/          # Trading volume factors
```

## Usage

```rust,ignore
use factors::{Factor, FactorRegistry, FactorCategory};

// Create registry with all default factors
let registry = FactorRegistry::with_defaults();

// Get all momentum factors
let momentum = registry.by_category(FactorCategory::Momentum);

// Compute a specific factor
let short_momentum = registry.get("short_term_momentum").unwrap();
let result = short_momentum.compute(&data, date)?;
```

## Factor Trait

All factors implement the core `Factor` trait:

```rust,ignore
pub trait Factor: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn category(&self) -> FactorCategory;
    fn required_columns(&self) -> &[&str];
    fn lookback(&self) -> usize;
    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame>;
    fn compute(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame>;
}
```

## Standardization

All factors support cross-sectional standardization:

- `cross_sectional_standardize`: Z-score normalization per date
- `winsorize`: Clip extreme values to percentiles
- `robust_standardize`: MAD-based standardization for outlier robustness

## License

MIT License - see [LICENSE](../../LICENSE).

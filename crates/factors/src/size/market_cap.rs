//! Market capitalization factor.
//!
//! Computes raw market capitalization (price × shares outstanding).
//! Unlike the log_mcap factor, this returns the raw value without logarithmic transformation.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for market capitalization factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct MarketCapConfig {
    // Empty for now - size factors typically don't need configuration
    // This struct exists for consistency and future extensibility
}

/// Market capitalization factor.
///
/// This factor measures company size using raw market capitalization.
/// Market cap represents the total market value of a company's outstanding shares.
///
/// # Formula
///
/// `market_cap = close × shares_outstanding`
///
/// # Properties
///
/// - **Category**: Size
/// - **Frequency**: Daily
/// - **Lookback**: 1 day (current values only)
/// - **Required columns**: `["symbol", "date", "close", "shares_outstanding"]`
///
/// # Example
///
/// ```ignore
/// use fd_factors::{Factor, size::MarketCap};
/// use polars::prelude::*;
/// use chrono::NaiveDate;
///
/// let factor = MarketCap::default();
/// let data = df![
///     "symbol" => ["AAPL", "MSFT"],
///     "date" => ["2024-01-01", "2024-01-01"],
///     "close" => [150.0, 300.0],
///     "shares_outstanding" => [16_000_000_000.0, 7_500_000_000.0],
/// ]?.lazy();
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let result = factor.compute(&data, date)?;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MarketCap {
    config: MarketCapConfig,
}

impl Factor for MarketCap {
    fn name(&self) -> &str {
        "market_cap"
    }

    fn description(&self) -> &str {
        "Raw market capitalization (price × shares outstanding)"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Size
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close", "shares_outstanding"]
    }

    fn lookback(&self) -> usize {
        1
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let date_str = date.format("%Y-%m-%d").to_string();

        let result = data
            .clone()
            .filter(col("date").eq(lit(date_str)))
            .select([
                col("symbol"),
                col("date"),
                (col("close") * col("shares_outstanding")).alias("market_cap"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for MarketCap {
    type Config = MarketCapConfig;

    fn with_config(config: Self::Config) -> Self {
        Self { config }
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_market_cap_computation() {
        // Create test data
        let df = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-01-01", "2024-01-01", "2024-01-01"],
            "close" => [150.0, 300.0, 100.0],
            "shares_outstanding" => [16_000_000_000.0, 7_500_000_000.0, 10_000_000_000.0],
        ]
        .unwrap();

        let factor = MarketCap::default();
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let result = factor.compute_raw(&df.lazy(), date).unwrap();

        // Extract results
        let symbols = result
            .column("symbol")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>();
        let mcaps = result
            .column("market_cap")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>();

        // Verify calculations
        // AAPL: 150 * 16e9 = 2.4e12
        // MSFT: 300 * 7.5e9 = 2.25e12
        // GOOGL: 100 * 10e9 = 1e12

        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0], "AAPL");
        assert_eq!(symbols[1], "MSFT");
        assert_eq!(symbols[2], "GOOGL");

        assert_relative_eq!(mcaps[0], 150.0 * 16_000_000_000.0_f64, epsilon = 1e-6);
        assert_relative_eq!(mcaps[1], 300.0 * 7_500_000_000.0_f64, epsilon = 1e-6);
        assert_relative_eq!(mcaps[2], 100.0 * 10_000_000_000.0_f64, epsilon = 1e-6);
    }

    #[test]
    fn test_market_cap_filters_by_date() {
        // Create test data with multiple dates
        let df = df![
            "symbol" => ["AAPL", "AAPL", "MSFT"],
            "date" => ["2024-01-01", "2024-01-02", "2024-01-01"],
            "close" => [150.0, 155.0, 300.0],
            "shares_outstanding" => [16_000_000_000.0, 16_000_000_000.0, 7_500_000_000.0],
        ]
        .unwrap();

        let factor = MarketCap::default();
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let result = factor.compute_raw(&df.lazy(), date).unwrap();

        // Should only return data for 2024-01-01
        assert_eq!(result.height(), 2);

        let symbols = result
            .column("symbol")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>();

        assert_eq!(symbols[0], "AAPL");
        assert_eq!(symbols[1], "MSFT");
    }

    #[test]
    fn test_market_cap_trait_properties() {
        let factor = MarketCap::default();

        assert_eq!(factor.name(), "market_cap");
        assert_eq!(
            factor.description(),
            "Raw market capitalization (price × shares outstanding)"
        );
        assert_eq!(factor.category(), FactorCategory::Size);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "close", "shares_outstanding"]
        );
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
    }

    #[test]
    fn test_market_cap_output_columns() {
        let df = df![
            "symbol" => ["AAPL"],
            "date" => ["2024-01-01"],
            "close" => [150.0],
            "shares_outstanding" => [16_000_000_000.0],
        ]
        .unwrap();

        let factor = MarketCap::default();
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let result = factor.compute_raw(&df.lazy(), date).unwrap();

        // Verify output schema
        let schema = result.schema();
        assert!(schema.contains("symbol"));
        assert!(schema.contains("date"));
        assert!(schema.contains("market_cap"));
        assert_eq!(schema.len(), 3);
    }
}

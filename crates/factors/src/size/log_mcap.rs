//! Log market capitalization factor.
//!
//! Computes the natural logarithm of market capitalization (price × shares outstanding).
//! The log transformation normalizes the distribution and makes the factor more suitable
//! for cross-sectional regression.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for log market capitalization factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogMarketCapConfig {
    // Empty for now - size factors typically don't need configuration
    // This struct exists for consistency and future extensibility
}

/// Log market capitalization factor.
///
/// This factor measures company size using the natural logarithm of market cap.
/// The log transformation ensures the factor has better statistical properties
/// for factor models, reducing the impact of extreme outliers.
///
/// # Formula
///
/// `log_market_cap = ln(close × shares_outstanding)`
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
/// use fd_factors::{Factor, size::LogMarketCap};
/// use polars::prelude::*;
/// use chrono::NaiveDate;
///
/// let factor = LogMarketCap::default();
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
pub struct LogMarketCap {
    config: LogMarketCapConfig,
}

impl Factor for LogMarketCap {
    fn name(&self) -> &str {
        "log_market_cap"
    }

    fn description(&self) -> &str {
        "Natural logarithm of market capitalization (price × shares outstanding)"
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
                (col("close") * col("shares_outstanding"))
                    .log(std::f64::consts::E)
                    .alias("log_market_cap"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for LogMarketCap {
    type Config = LogMarketCapConfig;

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
    fn test_log_market_cap_computation() {
        // Create test data
        let df = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-01-01", "2024-01-01", "2024-01-01"],
            "close" => [150.0, 300.0, 100.0],
            "shares_outstanding" => [16_000_000_000.0, 7_500_000_000.0, 10_000_000_000.0],
        ]
        .unwrap();

        let factor = LogMarketCap::default();
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
        let log_mcaps = result
            .column("log_market_cap")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>();

        // Verify calculations
        // AAPL: ln(150 * 16e9) = ln(2.4e12) ≈ 28.507
        // MSFT: ln(300 * 7.5e9) = ln(2.25e12) ≈ 28.442
        // GOOGL: ln(100 * 10e9) = ln(1e12) ≈ 27.631

        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0], "AAPL");
        assert_eq!(symbols[1], "MSFT");
        assert_eq!(symbols[2], "GOOGL");

        assert_relative_eq!(
            log_mcaps[0],
            (150.0 * 16_000_000_000.0_f64).ln(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            log_mcaps[1],
            (300.0 * 7_500_000_000.0_f64).ln(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            log_mcaps[2],
            (100.0 * 10_000_000_000.0_f64).ln(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_log_market_cap_filters_by_date() {
        // Create test data with multiple dates
        let df = df![
            "symbol" => ["AAPL", "AAPL", "MSFT"],
            "date" => ["2024-01-01", "2024-01-02", "2024-01-01"],
            "close" => [150.0, 155.0, 300.0],
            "shares_outstanding" => [16_000_000_000.0, 16_000_000_000.0, 7_500_000_000.0],
        ]
        .unwrap();

        let factor = LogMarketCap::default();
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
    fn test_log_market_cap_trait_properties() {
        let factor = LogMarketCap::default();

        assert_eq!(factor.name(), "log_market_cap");
        assert_eq!(
            factor.description(),
            "Natural logarithm of market capitalization (price × shares outstanding)"
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
    fn test_log_market_cap_output_columns() {
        let df = df![
            "symbol" => ["AAPL"],
            "date" => ["2024-01-01"],
            "close" => [150.0],
            "shares_outstanding" => [16_000_000_000.0],
        ]
        .unwrap();

        let factor = LogMarketCap::default();
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let result = factor.compute_raw(&df.lazy(), date).unwrap();

        // Verify output schema
        let schema = result.schema();
        assert!(schema.contains("symbol"));
        assert!(schema.contains("date"));
        assert!(schema.contains("log_market_cap"));
        assert_eq!(schema.len(), 3);
    }
}

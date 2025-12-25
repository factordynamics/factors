//! Earnings growth factor implementation.
//!
//! Measures year-over-year earnings per share (EPS) growth rate:
//! (EPS_t / EPS_{t-4}) - 1
//!
//! Companies with strong earnings growth may continue to outperform
//! as investors reward improving fundamentals.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for the Earnings Growth factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarningsGrowthConfig {
    /// Number of quarters to look back for growth calculation.
    /// Default is 4 (year-over-year). Use 2 for semi-annual, 8 for 2-year growth.
    pub growth_periods: usize,
}

impl Default for EarningsGrowthConfig {
    fn default() -> Self {
        Self { growth_periods: 4 }
    }
}

/// Earnings growth factor - year-over-year EPS growth rate.
///
/// Computes (EPS_t / EPS_{t-4}) - 1 using quarterly data.
/// Higher values indicate faster earnings growth.
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Reporting date (quarterly)
/// - `eps`: Earnings per share
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `earnings_growth`
#[derive(Debug, Clone, Default)]
pub struct EarningsGrowth {
    config: EarningsGrowthConfig,
}

impl Factor for EarningsGrowth {
    fn name(&self) -> &str {
        "earnings_growth"
    }

    fn description(&self) -> &str {
        "Year-over-year earnings per share growth rate: (EPS_t / EPS_{t-4}) - 1"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Growth
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "eps"]
    }

    fn lookback(&self) -> usize {
        self.config.growth_periods
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Quarterly
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Filter data up to the specified date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())))
            .collect()?;

        // Sort by symbol and date
        let sorted = filtered
            .lazy()
            .sort(
                ["symbol", "date"],
                SortMultipleOptions::default()
                    .with_order_descending_multi([false, false])
                    .with_nulls_last(true),
            )
            .collect()?;

        // Compute growth: (EPS_t / EPS_{t-n}) - 1
        let lag_alias = format!("eps_lag{}", self.config.growth_periods);
        let result = sorted
            .lazy()
            .with_column(
                col("eps")
                    .shift(lit(self.config.growth_periods as i64))
                    .over([col("symbol")])
                    .alias(&lag_alias),
            )
            .filter(col("date").eq(lit(date.to_string())))
            .with_column(((col("eps") / col(&lag_alias)) - lit(1.0)).alias("earnings_growth"))
            .select([col("symbol"), col("date"), col("earnings_growth")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for EarningsGrowth {
    type Config = EarningsGrowthConfig;

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

    #[test]
    fn test_earnings_growth() {
        // Create test data with quarterly EPS
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL",
                        "MSFT", "MSFT", "MSFT", "MSFT", "MSFT"],
            "date" => ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01",
                      "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
            "eps" => [1.0, 1.1, 1.2, 1.3, 1.4,
                     2.0, 2.2, 2.4, 2.6, 2.8]
        ]
        .unwrap();

        let factor = EarningsGrowth::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // AAPL: (1.4 / 1.0) - 1 = 0.4 (40% growth)
        // MSFT: (2.8 / 2.0) - 1 = 0.4 (40% growth)
        assert_eq!(result.height(), 2);
        assert!(result.column("earnings_growth").is_ok());

        let growth = result.column("earnings_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() - 0.4).abs() < 0.01);
        assert!((growth.get(1).unwrap() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_earnings_growth_metadata() {
        let factor = EarningsGrowth::default();
        assert_eq!(factor.name(), "earnings_growth");
        assert_eq!(factor.category(), FactorCategory::Growth);
        assert_eq!(factor.lookback(), 4);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.required_columns(), &["symbol", "date", "eps"]);
    }

    #[test]
    fn test_earnings_growth_custom_period() {
        // Test with 2-quarter (semi-annual) growth
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL"],
            "date" => ["2023-07-01", "2023-10-01", "2024-01-01"],
            "eps" => [1.0, 1.1, 1.2]
        ]
        .unwrap();

        let config = EarningsGrowthConfig { growth_periods: 2 };
        let factor = EarningsGrowth::with_config(config);
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // (1.2 / 1.0) - 1 = 0.2 (20% growth over 2 quarters)
        assert_eq!(result.height(), 1);
        let growth = result.column("earnings_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() - 0.2).abs() < 0.01);
    }
}

//! Sales growth factor implementation.
//!
//! Measures year-over-year revenue growth rate:
//! (Revenue_t / Revenue_{t-4}) - 1
//!
//! Companies with strong sales growth may continue to outperform
//! as increasing revenues signal market demand and competitive strength.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for the Sales Growth factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalesGrowthConfig {
    /// Number of quarters to look back for growth calculation.
    /// Default is 4 (year-over-year). Use 2 for semi-annual, 8 for 2-year growth.
    pub growth_periods: usize,
}

impl Default for SalesGrowthConfig {
    fn default() -> Self {
        Self { growth_periods: 4 }
    }
}

/// Sales growth factor - year-over-year revenue growth rate.
///
/// Computes (Revenue_t / Revenue_{t-4}) - 1 using quarterly data.
/// Higher values indicate faster sales growth.
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Reporting date (quarterly)
/// - `revenue`: Total revenue
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `sales_growth`
#[derive(Debug, Clone, Default)]
pub struct SalesGrowth {
    config: SalesGrowthConfig,
}

impl Factor for SalesGrowth {
    fn name(&self) -> &str {
        "sales_growth"
    }

    fn description(&self) -> &str {
        "Year-over-year revenue growth rate: (Revenue_t / Revenue_{t-4}) - 1"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Growth
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "revenue"]
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

        // Compute growth: (Revenue_t / Revenue_{t-n}) - 1
        let lag_alias = format!("revenue_lag{}", self.config.growth_periods);
        let result = sorted
            .lazy()
            .with_column(
                col("revenue")
                    .shift(lit(self.config.growth_periods as i64))
                    .over([col("symbol")])
                    .alias(&lag_alias),
            )
            .filter(col("date").eq(lit(date.to_string())))
            .with_column(((col("revenue") / col(&lag_alias)) - lit(1.0)).alias("sales_growth"))
            .select([col("symbol"), col("date"), col("sales_growth")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for SalesGrowth {
    type Config = SalesGrowthConfig;

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
    fn test_sales_growth() {
        // Create test data with quarterly revenue
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL",
                        "MSFT", "MSFT", "MSFT", "MSFT", "MSFT"],
            "date" => ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01",
                      "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
            "revenue" => [100.0, 110.0, 120.0, 130.0, 150.0,
                         200.0, 220.0, 240.0, 260.0, 280.0]
        ]
        .unwrap();

        let factor = SalesGrowth::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // AAPL: (150 / 100) - 1 = 0.5 (50% growth)
        // MSFT: (280 / 200) - 1 = 0.4 (40% growth)
        assert_eq!(result.height(), 2);
        assert!(result.column("sales_growth").is_ok());

        let growth = result.column("sales_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() - 0.5).abs() < 0.01);
        assert!((growth.get(1).unwrap() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_sales_growth_metadata() {
        let factor = SalesGrowth::default();
        assert_eq!(factor.name(), "sales_growth");
        assert_eq!(factor.category(), FactorCategory::Growth);
        assert_eq!(factor.lookback(), 4);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.required_columns(), &["symbol", "date", "revenue"]);
    }
}

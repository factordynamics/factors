//! Asset growth factor implementation.
//!
//! Measures year-over-year total assets growth rate:
//! (Total Assets_t / Total Assets_{t-4}) - 1
//!
//! Academic: Cooper, Gulen, Schill (2008) - firms with high asset growth
//! tend to underperform, potentially due to overinvestment or empire building.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Asset growth factor - year-over-year total assets growth rate.
///
/// Computes (Total Assets_t / Total Assets_{t-4}) - 1 using quarterly data.
/// Higher values indicate faster asset expansion.
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Reporting date (quarterly)
/// - `total_assets`: Total assets
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `asset_growth`
#[derive(Debug, Clone, Default)]
pub struct AssetGrowth;

impl Factor for AssetGrowth {
    fn name(&self) -> &str {
        "asset_growth"
    }

    fn description(&self) -> &str {
        "Year-over-year total assets growth rate: (Total Assets_t / Total Assets_{t-4}) - 1"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Growth
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "total_assets"]
    }

    fn lookback(&self) -> usize {
        4 // 4 quarters for year-over-year comparison
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

        // Compute year-over-year growth: (Total Assets_t / Total Assets_{t-4}) - 1
        let result = sorted
            .lazy()
            .with_column(
                col("total_assets")
                    .shift(lit(4))
                    .over([col("symbol")])
                    .alias("total_assets_lag4"),
            )
            .filter(col("date").eq(lit(date.to_string())))
            .with_column(
                ((col("total_assets") / col("total_assets_lag4")) - lit(1.0)).alias("asset_growth"),
            )
            .select([col("symbol"), col("date"), col("asset_growth")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_growth() {
        // Create test data with quarterly total assets
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL",
                        "MSFT", "MSFT", "MSFT", "MSFT", "MSFT"],
            "date" => ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01",
                      "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
            "total_assets" => [100.0, 105.0, 110.0, 115.0, 120.0,
                              200.0, 210.0, 220.0, 230.0, 240.0]
        ]
        .unwrap();

        let factor = AssetGrowth;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // AAPL: (120 / 100) - 1 = 0.2 (20% growth)
        // MSFT: (240 / 200) - 1 = 0.2 (20% growth)
        assert_eq!(result.height(), 2);
        assert!(result.column("asset_growth").is_ok());

        let growth = result.column("asset_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() - 0.2).abs() < 0.01);
        assert!((growth.get(1).unwrap() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_asset_growth_metadata() {
        let factor = AssetGrowth;
        assert_eq!(factor.name(), "asset_growth");
        assert_eq!(factor.category(), FactorCategory::Growth);
        assert_eq!(factor.lookback(), 4);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "total_assets"]
        );
    }

    #[test]
    fn test_asset_growth_negative() {
        // Test with declining assets
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "date" => ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
            "total_assets" => [100.0, 95.0, 90.0, 85.0, 80.0]
        ]
        .unwrap();

        let factor = AssetGrowth;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // AAPL: (80 / 100) - 1 = -0.2 (-20% decline)
        assert_eq!(result.height(), 1);
        let growth = result.column("asset_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() + 0.2).abs() < 0.01);
    }
}

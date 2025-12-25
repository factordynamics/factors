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
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

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
pub struct SalesGrowth;

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

        // Compute year-over-year growth: (Revenue_t / Revenue_{t-4}) - 1
        let result = sorted
            .lazy()
            .with_column(
                col("revenue")
                    .shift(lit(4))
                    .over([col("symbol")])
                    .alias("revenue_lag4"),
            )
            .filter(col("date").eq(lit(date.to_string())))
            .with_column(((col("revenue") / col("revenue_lag4")) - lit(1.0)).alias("sales_growth"))
            .select([col("symbol"), col("date"), col("sales_growth")])
            .collect()?;

        Ok(result)
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

        let factor = SalesGrowth;
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
        let factor = SalesGrowth;
        assert_eq!(factor.name(), "sales_growth");
        assert_eq!(factor.category(), FactorCategory::Growth);
        assert_eq!(factor.lookback(), 4);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.required_columns(), &["symbol", "date", "revenue"]);
    }
}

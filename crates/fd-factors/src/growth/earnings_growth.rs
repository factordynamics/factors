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
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

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
pub struct EarningsGrowth;

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

        // Compute year-over-year growth: (EPS_t / EPS_{t-4}) - 1
        let result = sorted
            .lazy()
            .with_column(
                col("eps")
                    .shift(lit(4))
                    .over([col("symbol")])
                    .alias("eps_lag4"),
            )
            .filter(col("date").eq(lit(date.to_string())))
            .with_column(((col("eps") / col("eps_lag4")) - lit(1.0)).alias("earnings_growth"))
            .select([col("symbol"), col("date"), col("earnings_growth")])
            .collect()?;

        Ok(result)
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

        let factor = EarningsGrowth;
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
        let factor = EarningsGrowth;
        assert_eq!(factor.name(), "earnings_growth");
        assert_eq!(factor.category(), FactorCategory::Growth);
        assert_eq!(factor.lookback(), 4);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.required_columns(), &["symbol", "date", "eps"]);
    }
}

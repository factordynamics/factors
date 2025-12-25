//! Book equity growth factor implementation.
//!
//! Measures year-over-year book equity growth rate:
//! (Book Equity_t / Book Equity_{t-4}) - 1
//!
//! Book equity growth captures changes in shareholder equity from
//! retained earnings and equity issuance.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Book equity growth factor - year-over-year book equity growth rate.
///
/// Computes (Book Equity_t / Book Equity_{t-4}) - 1 using quarterly data.
/// Higher values indicate faster equity base expansion.
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Reporting date (quarterly)
/// - `book_equity`: Book value of equity
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `book_equity_growth`
#[derive(Debug, Clone, Default)]
pub struct BookEquityGrowth;

impl Factor for BookEquityGrowth {
    fn name(&self) -> &str {
        "book_equity_growth"
    }

    fn description(&self) -> &str {
        "Year-over-year book equity growth rate: (Book Equity_t / Book Equity_{t-4}) - 1"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Growth
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "book_equity"]
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

        // Compute year-over-year growth: (Book Equity_t / Book Equity_{t-4}) - 1
        let result = sorted
            .lazy()
            .with_column(
                col("book_equity")
                    .shift(lit(4))
                    .over([col("symbol")])
                    .alias("book_equity_lag4"),
            )
            .filter(col("date").eq(lit(date.to_string())))
            .with_column(
                ((col("book_equity") / col("book_equity_lag4")) - lit(1.0))
                    .alias("book_equity_growth"),
            )
            .select([col("symbol"), col("date"), col("book_equity_growth")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_equity_growth() {
        // Create test data with quarterly book equity
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL",
                        "MSFT", "MSFT", "MSFT", "MSFT", "MSFT"],
            "date" => ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01",
                      "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
            "book_equity" => [50.0, 52.0, 54.0, 56.0, 60.0,
                             100.0, 105.0, 110.0, 115.0, 125.0]
        ]
        .unwrap();

        let factor = BookEquityGrowth;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // AAPL: (60 / 50) - 1 = 0.2 (20% growth)
        // MSFT: (125 / 100) - 1 = 0.25 (25% growth)
        assert_eq!(result.height(), 2);
        assert!(result.column("book_equity_growth").is_ok());

        let growth = result.column("book_equity_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() - 0.2).abs() < 0.01);
        assert!((growth.get(1).unwrap() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_book_equity_growth_metadata() {
        let factor = BookEquityGrowth;
        assert_eq!(factor.name(), "book_equity_growth");
        assert_eq!(factor.category(), FactorCategory::Growth);
        assert_eq!(factor.lookback(), 4);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "book_equity"]
        );
    }

    #[test]
    fn test_book_equity_growth_decline() {
        // Test with declining book equity (e.g., from losses or buybacks)
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "date" => ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
            "book_equity" => [100.0, 95.0, 90.0, 85.0, 75.0]
        ]
        .unwrap();

        let factor = BookEquityGrowth;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // AAPL: (75 / 100) - 1 = -0.25 (-25% decline)
        assert_eq!(result.height(), 1);
        let growth = result.column("book_equity_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() + 0.25).abs() < 0.01);
    }
}

//! Institutional Ownership Change factor - Quarterly change in institutional holdings.
//!
//! This factor measures the change in the percentage of shares held by
//! institutional investors. Increasing institutional ownership can signal
//! improving fundamentals and provide price support, while decreasing
//! ownership may indicate deteriorating prospects.
//!
//! # Academic Foundation
//! Gompers & Metrick (2001) - "Institutional Investors and Equity Prices"
//! Documents that stocks with increasing institutional ownership outperform,
//! partly due to institutional demand pressure and their superior information.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Institutional Ownership Change factor.
///
/// Computes the quarterly change in institutional ownership percentage:
/// `IO_t - IO_{t-1}`
///
/// where:
/// - `IO_t` is the current institutional ownership percentage
/// - `IO_{t-1}` is the prior quarter's institutional ownership percentage
///
/// Positive values indicate increasing institutional interest (bullish),
/// while negative values indicate decreasing interest (bearish).
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Quarter end date or filing date
/// - `institutional_ownership`: Percentage of shares held by institutions (0-100)
///
/// # Lookback Period
/// 1 quarter (need current and prior quarter data)
///
/// # Usage Notes
/// - Data typically comes from 13-F filings (filed within 45 days of quarter end)
/// - Focus on meaningful changes (e.g., > 2 percentage points)
/// - Combines well with momentum and quality factors
/// - Large cap stocks tend to have higher institutional ownership
/// - Extreme values may indicate index inclusion/exclusion events
#[derive(Debug, Clone, Default)]
pub struct InstitutionalOwnership;

impl Factor for InstitutionalOwnership {
    fn name(&self) -> &str {
        "institutional_ownership_change"
    }

    fn description(&self) -> &str {
        "Quarterly change in institutional ownership percentage"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Sentiment
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "institutional_ownership"]
    }

    fn lookback(&self) -> usize {
        1 // Need 1 prior quarter
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Quarterly
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Filter data up to the target date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.format("%Y-%m-%d").to_string())))
            .collect()?;

        // Group by symbol and compute ownership change
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                col("institutional_ownership")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("current_ownership"),
                col("institutional_ownership")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit(self.lookback() as i64 + 1)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("prior_ownership"),
            ])
            .filter(col("prior_ownership").is_not_null()) // Need both quarters - filter before calculation
            .with_column(
                // Calculate ownership change: current - prior
                (col("current_ownership") - col("prior_ownership")).alias(self.name()),
            )
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    #[test]
    fn test_institutional_ownership_increase() {
        let factor = InstitutionalOwnership;

        // Create test data with increasing ownership from 60% to 65%
        let df = df! {
            "symbol" => vec!["AAPL", "AAPL"],
            "date" => vec!["2023-12-31", "2024-03-31"],
            "institutional_ownership" => vec![60.0, 65.0],
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Change = 65.0 - 60.0 = 5.0
        let change = result
            .column("institutional_ownership_change")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (change - 5.0).abs() < 0.01,
            "Expected change of 5.0, got {}",
            change
        );
    }

    #[test]
    fn test_institutional_ownership_decrease() {
        let factor = InstitutionalOwnership;

        // Create test data with decreasing ownership from 70% to 65%
        let df = df! {
            "symbol" => vec!["MSFT", "MSFT"],
            "date" => vec!["2023-12-31", "2024-03-31"],
            "institutional_ownership" => vec![70.0, 65.0],
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // Change = 65.0 - 70.0 = -5.0
        let change = result
            .column("institutional_ownership_change")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (change + 5.0).abs() < 0.01,
            "Expected change of -5.0, got {}",
            change
        );
    }

    #[test]
    fn test_institutional_ownership_multiple_stocks() {
        let factor = InstitutionalOwnership;

        // Create test data for multiple stocks
        let df = df! {
            "symbol" => vec!["AAPL", "AAPL", "MSFT", "MSFT", "GOOGL", "GOOGL"],
            "date" => vec![
                "2023-12-31", "2024-03-31",
                "2023-12-31", "2024-03-31",
                "2023-12-31", "2024-03-31"
            ],
            "institutional_ownership" => vec![
                60.0, 65.0,  // AAPL: +5
                70.0, 68.0,  // MSFT: -2
                55.0, 55.0,  // GOOGL: 0
            ],
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 3);
        assert!(result.column("institutional_ownership_change").is_ok());

        let syms = result.column("symbol").unwrap().str().unwrap();
        let changes = result
            .column("institutional_ownership_change")
            .unwrap()
            .f64()
            .unwrap();

        for i in 0..result.height() {
            let sym = syms.get(i).unwrap();
            let change = changes.get(i).unwrap();

            match sym {
                "AAPL" => assert!(
                    (change - 5.0).abs() < 0.01,
                    "AAPL change should be 5.0, got {}",
                    change
                ),
                "MSFT" => assert!(
                    (change + 2.0).abs() < 0.01,
                    "MSFT change should be -2.0, got {}",
                    change
                ),
                "GOOGL" => assert!(
                    change.abs() < 0.01,
                    "GOOGL change should be 0.0, got {}",
                    change
                ),
                _ => panic!("Unexpected symbol: {}", sym),
            }
        }
    }

    #[test]
    fn test_institutional_ownership_metadata() {
        let factor = InstitutionalOwnership;

        assert_eq!(factor.name(), "institutional_ownership_change");
        assert_eq!(factor.category(), FactorCategory::Sentiment);
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "institutional_ownership"]
        );
    }

    #[test]
    fn test_institutional_ownership_single_quarter() {
        let factor = InstitutionalOwnership;

        // Create test data with only one quarter (insufficient for change calculation)
        let df = df! {
            "symbol" => vec!["AAPL"],
            "date" => vec!["2024-03-31"],
            "institutional_ownership" => vec![65.0],
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        // Should return empty due to missing prior quarter
        assert_eq!(result.height(), 0);
    }

    #[test]
    fn test_institutional_ownership_multiple_quarters() {
        let factor = InstitutionalOwnership;

        // Create test data with 4 quarters - should use most recent change
        let df = df! {
            "symbol" => vec!["AAPL", "AAPL", "AAPL", "AAPL"],
            "date" => vec!["2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"],
            "institutional_ownership" => vec![55.0, 58.0, 60.0, 65.0],
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // Should calculate change from most recent two quarters: 65.0 - 60.0 = 5.0
        let change = result
            .column("institutional_ownership_change")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (change - 5.0).abs() < 0.01,
            "Expected change of 5.0 (most recent quarter), got {}",
            change
        );
    }
}

//! Earnings Surprise (SUE) factor - Standardized Unexpected Earnings.
//!
//! This factor measures the standardized difference between actual and expected
//! earnings per share (EPS). Stocks with positive earnings surprises tend to
//! outperform due to post-earnings-announcement drift (PEAD).
//!
//! # Academic Foundation
//! Bernard & Thomas (1989) - "Post-Earnings-Announcement Drift: Delayed Price
//! Response or Risk Premium?" Documents that stock prices do not fully adjust
//! to earnings surprises immediately, creating predictable drift.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Earnings Surprise factor measuring standardized unexpected earnings.
///
/// Computes the standardized earnings surprise:
/// `(Actual EPS - Expected EPS) / σ(surprise)`
///
/// where:
/// - `Actual EPS` is the reported earnings per share
/// - `Expected EPS` is the consensus analyst estimate
/// - `σ(surprise)` is the standard deviation of the surprise
///
/// Higher values indicate larger positive surprises, which tend to be followed
/// by positive drift in stock prices over subsequent quarters.
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Earnings announcement date
/// - `eps_actual`: Actual reported EPS
/// - `eps_expected`: Consensus expected EPS
/// - `surprise_std`: Standard deviation of earnings surprise
///
/// # Usage
/// This is a quarterly factor, typically computed on earnings announcement dates.
/// When `surprise_std` is not available, you can use a simple unstandardized
/// version: `(eps_actual - eps_expected) / |eps_expected|`
#[derive(Debug, Clone, Default)]
pub struct EarningsSurprise;

impl Factor for EarningsSurprise {
    fn name(&self) -> &str {
        "earnings_surprise"
    }

    fn description(&self) -> &str {
        "Standardized Unexpected Earnings (SUE) - Post-Earnings-Announcement Drift"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Sentiment
    }

    fn required_columns(&self) -> &[&str] {
        &[
            "symbol",
            "date",
            "eps_actual",
            "eps_expected",
            "surprise_std",
        ]
    }

    fn lookback(&self) -> usize {
        1 // Only need current quarter's data
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

        // For each symbol, get the most recent earnings announcement
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                col("eps_actual")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("eps_actual"),
                col("eps_expected")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("eps_expected"),
                col("surprise_std")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("surprise_std"),
            ])
            .filter(col("surprise_std").gt(lit(0.0))) // Filter out zero std dev before calculation
            .with_column(
                // Calculate SUE: (actual - expected) / std(surprise)
                ((col("eps_actual") - col("eps_expected")) / col("surprise_std"))
                    .alias(self.name()),
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
    fn test_earnings_surprise_positive() {
        let factor = EarningsSurprise;

        // Create test data with positive surprise
        let df = df! {
            "symbol" => vec!["AAPL"],
            "date" => vec!["2024-01-15"],
            "eps_actual" => vec![2.5],
            "eps_expected" => vec![2.0],
            "surprise_std" => vec![0.25],
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // SUE = (2.5 - 2.0) / 0.25 = 2.0
        let sue = result
            .column("earnings_surprise")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!((sue - 2.0).abs() < 0.01, "Expected SUE of 2.0, got {}", sue);
    }

    #[test]
    fn test_earnings_surprise_negative() {
        let factor = EarningsSurprise;

        // Create test data with negative surprise
        let df = df! {
            "symbol" => vec!["MSFT"],
            "date" => vec!["2024-01-15"],
            "eps_actual" => vec![1.5],
            "eps_expected" => vec![2.0],
            "surprise_std" => vec![0.5],
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // SUE = (1.5 - 2.0) / 0.5 = -1.0
        let sue = result
            .column("earnings_surprise")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (sue + 1.0).abs() < 0.01,
            "Expected SUE of -1.0, got {}",
            sue
        );
    }

    #[test]
    fn test_earnings_surprise_multiple_stocks() {
        let factor = EarningsSurprise;

        // Create test data for multiple stocks
        let df = df! {
            "symbol" => vec!["AAPL", "MSFT", "GOOGL"],
            "date" => vec!["2024-01-15", "2024-01-16", "2024-01-17"],
            "eps_actual" => vec![2.5, 3.0, 1.8],
            "eps_expected" => vec![2.0, 3.0, 2.0],
            "surprise_std" => vec![0.25, 0.5, 0.1],
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 3);
        assert!(result.column("earnings_surprise").is_ok());

        let symbols = result.column("symbol").unwrap().str().unwrap();
        let sues = result.column("earnings_surprise").unwrap().f64().unwrap();

        // AAPL: (2.5 - 2.0) / 0.25 = 2.0
        // MSFT: (3.0 - 3.0) / 0.5 = 0.0
        // GOOGL: (1.8 - 2.0) / 0.1 = -2.0
        for i in 0..result.height() {
            let sym = symbols.get(i).unwrap();
            let sue = sues.get(i).unwrap();

            match sym {
                "AAPL" => assert!((sue - 2.0).abs() < 0.01),
                "MSFT" => assert!(sue.abs() < 0.01),
                "GOOGL" => assert!((sue + 2.0).abs() < 0.01),
                _ => panic!("Unexpected symbol: {}", sym),
            }
        }
    }

    #[test]
    fn test_earnings_surprise_metadata() {
        let factor = EarningsSurprise;

        assert_eq!(factor.name(), "earnings_surprise");
        assert_eq!(factor.category(), FactorCategory::Sentiment);
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(
            factor.required_columns(),
            &[
                "symbol",
                "date",
                "eps_actual",
                "eps_expected",
                "surprise_std"
            ]
        );
    }

    #[test]
    fn test_earnings_surprise_filters_zero_std() {
        let factor = EarningsSurprise;

        // Create test data with zero std dev (should be filtered out)
        let df = df! {
            "symbol" => vec!["AAPL", "MSFT"],
            "date" => vec!["2024-01-15", "2024-01-16"],
            "eps_actual" => vec![2.5, 3.0],
            "eps_expected" => vec![2.0, 3.0],
            "surprise_std" => vec![0.0, 0.5], // AAPL has zero std
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 31).unwrap())
            .unwrap();

        // Should only return MSFT (AAPL filtered due to zero std)
        assert_eq!(result.height(), 1);

        let sym = result
            .column("symbol")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap();

        assert_eq!(sym, "MSFT");
    }
}

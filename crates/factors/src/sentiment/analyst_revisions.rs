//! Analyst Revisions factor - measures net direction of analyst estimate changes.
//!
//! This factor captures the momentum in analyst earnings estimates. Stocks with
//! more positive revisions tend to outperform as estimates reflect improving
//! fundamentals and market sentiment.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for the Analyst Revisions factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalystRevisionsConfig {
    /// Number of days to look back for estimate comparison.
    /// Default is 63 (approximately 3 months). Use 21 for 1-month, 126 for 6-month changes.
    pub lookback_days: usize,
}

impl Default for AnalystRevisionsConfig {
    fn default() -> Self {
        Self { lookback_days: 63 }
    }
}

/// Analyst Revisions factor measuring the direction of EPS estimate changes.
///
/// Computes the percentage change in consensus EPS estimates:
/// `(Current EPS Estimate - Prior EPS Estimate) / |Prior EPS Estimate|`
///
/// where:
/// - `Current EPS Estimate` is the most recent consensus estimate
/// - `Prior EPS Estimate` is the estimate from lookback_days trading days ago
///
/// Positive values indicate upward revisions, negative values indicate downgrades.
/// This factor is useful for:
/// - Identifying improving fundamentals before they're reflected in prices
/// - Capturing analyst sentiment momentum
/// - Combining with momentum and value factors for enhanced alpha
///
/// # Alternative Formulation
/// When revision count data is available, can use:
/// `(Upward Revisions - Downward Revisions) / Total Revisions`
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Date of the estimate
/// - `eps_estimate`: Consensus EPS estimate
///
/// Or alternatively:
/// - `symbol`: Stock ticker symbol
/// - `date`: Date
/// - `revisions_up`: Number of upward revisions
/// - `revisions_down`: Number of downward revisions
#[derive(Debug, Clone, Default)]
pub struct AnalystRevisions {
    config: AnalystRevisionsConfig,
}

impl Factor for AnalystRevisions {
    fn name(&self) -> &str {
        "analyst_revisions"
    }

    fn description(&self) -> &str {
        "Analyst revision momentum - net direction of EPS estimate changes"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Sentiment
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "eps_estimate"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback_days
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Filter data up to the target date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.format("%Y-%m-%d").to_string())))
            .collect()?;

        // Group by symbol and compute revision momentum
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                col("eps_estimate")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("current_estimate"),
                col("eps_estimate")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit(self.config.lookback_days as i64 + 1)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("prior_estimate"),
            ])
            .with_column(
                // Calculate percentage change: (current - prior) / |prior|
                ((col("current_estimate") - col("prior_estimate")) / col("prior_estimate").abs())
                    .alias(self.name()),
            )
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for AnalystRevisions {
    type Config = AnalystRevisionsConfig;

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
    use polars::df;

    #[test]
    fn test_analyst_revisions_positive() {
        let factor = AnalystRevisions::default();

        // Create test data with 64 days of EPS estimates
        let dates: Vec<String> = (0..64)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 64];
        // EPS estimate increases from $5.00 to $5.50 (10% upward revision)
        let estimates: Vec<f64> = (0..64).map(|i| 5.0 + (i as f64 * 0.5 / 63.0)).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "eps_estimate" => estimates,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 4).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that revision is approximately 10% (0.10)
        let revision = result
            .column("analyst_revisions")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (revision - 0.10).abs() < 0.01,
            "Expected ~0.10, got {}",
            revision
        );
    }

    #[test]
    fn test_analyst_revisions_negative() {
        let factor = AnalystRevisions::default();

        // Create test data with 64 days of EPS estimates
        let dates: Vec<String> = (0..64)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 64];
        // EPS estimate decreases from $5.00 to $4.50 (10% downward revision)
        let estimates: Vec<f64> = (0..64).map(|i| 5.0 - (i as f64 * 0.5 / 63.0)).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "eps_estimate" => estimates,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 4).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // Check that revision is approximately -10% (-0.10)
        let revision = result
            .column("analyst_revisions")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (revision + 0.10).abs() < 0.01,
            "Expected ~-0.10, got {}",
            revision
        );
    }

    #[test]
    fn test_analyst_revisions_multiple_stocks() {
        let factor = AnalystRevisions::default();

        // Create test data for two stocks
        let mut dates = Vec::new();
        let mut symbols = Vec::new();
        let mut estimates = Vec::new();

        for _ in 0..64 {
            for (i, sym) in ["AAPL", "MSFT"].iter().enumerate() {
                dates.push(
                    NaiveDate::from_ymd_opt(2024, 1, 1)
                        .unwrap()
                        .checked_add_days(chrono::Days::new((dates.len() / 2) as u64))
                        .unwrap()
                        .format("%Y-%m-%d")
                        .to_string(),
                );
                symbols.push(*sym);
                // AAPL: increasing estimate, MSFT: flat estimate
                if i == 0 {
                    estimates.push(5.0 + (dates.len() as f64 / 2.0 * 0.5 / 63.0));
                } else {
                    estimates.push(3.0);
                }
            }
        }

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "eps_estimate" => estimates,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 4).unwrap())
            .unwrap();

        assert_eq!(result.height(), 2);
        assert!(result.column("analyst_revisions").is_ok());

        // AAPL should have positive revision, MSFT should have ~0
        let revisions = result.column("analyst_revisions").unwrap().f64().unwrap();
        let symbols_result = result.column("symbol").unwrap().str().unwrap();

        for i in 0..result.height() {
            let sym = symbols_result.get(i).unwrap();
            let rev = revisions.get(i).unwrap();

            if sym == "AAPL" {
                assert!(
                    rev > 0.05,
                    "AAPL should have positive revision, got {}",
                    rev
                );
            } else if sym == "MSFT" {
                assert!(
                    rev.abs() < 0.01,
                    "MSFT should have ~0 revision, got {}",
                    rev
                );
            }
        }
    }

    #[test]
    fn test_analyst_revisions_metadata() {
        let factor = AnalystRevisions::default();

        assert_eq!(factor.name(), "analyst_revisions");
        assert_eq!(factor.category(), FactorCategory::Sentiment);
        assert_eq!(factor.lookback(), 63);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "eps_estimate"]
        );
    }

    #[test]
    fn test_analyst_revisions_insufficient_history() {
        let factor = AnalystRevisions::default();

        // Create test data with only 30 days (insufficient for 63-day lookback)
        let dates: Vec<String> = (0..30)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 30];
        let estimates: Vec<f64> = (0..30).map(|i| 5.0 + (i as f64 * 0.1)).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "eps_estimate" => estimates,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 30).unwrap())
            .unwrap();

        // Should return empty or no results due to insufficient history
        assert_eq!(result.height(), 0);
    }
}

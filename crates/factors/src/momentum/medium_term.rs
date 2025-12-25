//! Medium-term momentum factor (6-month lookback).

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for medium-term momentum factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MediumTermMomentumConfig {
    /// Number of trading days to look back (default: 126)
    pub lookback: usize,
    /// Number of recent days to skip to avoid reversal effects (default: 21)
    pub skip_days: usize,
}

impl Default for MediumTermMomentumConfig {
    fn default() -> Self {
        Self {
            lookback: 126,
            skip_days: 21,
        }
    }
}

/// Medium-term momentum factor measuring 6-month (126 trading days) price change.
///
/// Measures the percentage price change over the past 6 months:
/// `(P_{t-skip} / P_{t-lookback-skip}) - 1`
///
/// where:
/// - `P_{t-skip}` is the price after skipping recent days
/// - `P_{t-lookback-skip}` is the price from lookback + skip days ago
///
/// The skip period helps avoid short-term reversal effects.
///
/// Captures medium-term trend persistence, useful for:
/// - Traditional momentum strategies
/// - Identifying sustained trends
/// - Risk model construction
///
/// This is a widely used momentum horizon that balances responsiveness
/// with stability, avoiding very short-term noise and capturing
/// established trends.
#[derive(Debug, Clone, Default)]
pub struct MediumTermMomentum {
    config: MediumTermMomentumConfig,
}

impl Factor for MediumTermMomentum {
    fn name(&self) -> &str {
        "medium_term_momentum"
    }

    fn description(&self) -> &str {
        "6-month (126-day) momentum - medium-term trend persistence"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback
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

        let skip_days = self.config.skip_days;
        let lookback = self.config.lookback;

        // Group by symbol and compute momentum with skip period
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                // Price from skip_days ago
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit(skip_days as i64 + 1)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("current_price"),
                // Price from lookback + skip_days ago
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit((lookback + skip_days) as i64 + 1)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("lagged_price"),
            ])
            .with_column(
                ((col("current_price") / col("lagged_price")) - lit(1.0)).alias(self.name()),
            )
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for MediumTermMomentum {
    type Config = MediumTermMomentumConfig;

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
    fn test_medium_term_momentum_basic() {
        let factor = MediumTermMomentum::default();

        // Create test data with 148 days of prices (lookback 126 + skip 21 + 1)
        let dates: Vec<String> = (0..148)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 148];
        // Price goes from 100 to 125 (25% gain)
        // At t-147 (index 0): price = 100.0
        // At t-21 (index 126): price = 100.0 + 126 * 0.1984 = 125.0
        // Momentum = (125.0 / 100.0) - 1 = 0.25 (25%)
        let prices: Vec<f64> = (0..148).map(|i| 100.0 + i as f64 * 0.1984).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 5, 28).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that momentum is approximately 25% (0.25)
        let momentum = result
            .column("medium_term_momentum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (momentum - 0.25).abs() < 0.01,
            "Expected ~0.25, got {}",
            momentum
        );
    }

    #[test]
    fn test_medium_term_momentum_metadata() {
        let factor = MediumTermMomentum::default();

        assert_eq!(factor.name(), "medium_term_momentum");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 126);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

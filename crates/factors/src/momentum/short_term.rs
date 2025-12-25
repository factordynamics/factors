//! Short-term momentum factor (1-month lookback).

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for short-term momentum factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShortTermMomentumConfig {
    /// Number of trading days to look back (default: 21)
    pub lookback: usize,
    /// Number of recent days to skip to avoid reversal effects (default: 5)
    pub skip_days: usize,
}

impl Default for ShortTermMomentumConfig {
    fn default() -> Self {
        Self {
            lookback: 21,
            skip_days: 5,
        }
    }
}

/// Short-term momentum factor measuring 1-month (21 trading days) price change.
///
/// Measures the percentage price change over the past month:
/// `(P_{t-skip} / P_{t-lookback-skip}) - 1`
///
/// where:
/// - `P_{t-skip}` is the price after skipping recent days
/// - `P_{t-lookback-skip}` is the price from lookback + skip days ago
///
/// The skip period helps avoid short-term reversal effects.
///
/// Captures short-term trend persistence, useful for:
/// - High-frequency rebalancing strategies
/// - Identifying recent price breakouts
/// - Combining with longer-term momentum for multi-scale strategies
#[derive(Debug, Clone, Default)]
pub struct ShortTermMomentum {
    config: ShortTermMomentumConfig,
}

impl Factor for ShortTermMomentum {
    fn name(&self) -> &str {
        "short_term_momentum"
    }

    fn description(&self) -> &str {
        "1-month (21-day) momentum - short-term trend persistence"
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

impl ConfigurableFactor for ShortTermMomentum {
    type Config = ShortTermMomentumConfig;

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
    fn test_short_term_momentum_basic() {
        let factor = ShortTermMomentum::default();

        // Create test data with 27 days of prices (lookback 21 + skip 5 + 1)
        let dates: Vec<String> = (0..27)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 27];
        // Price goes from 100 to 110 (10% gain)
        // At t-26 (index 0): price = 100.0
        // At t-5 (index 21): price = 100.0 + 21 * 0.476 = 110.0
        // Momentum = (110.0 / 100.0) - 1 = 0.10 (10%)
        let prices: Vec<f64> = (0..27).map(|i| 100.0 + i as f64 * 0.476).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 27).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that momentum is approximately 10% (0.10)
        let momentum = result
            .column("short_term_momentum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (momentum - 0.10).abs() < 0.01,
            "Expected ~0.10, got {}",
            momentum
        );
    }

    #[test]
    fn test_short_term_momentum_metadata() {
        let factor = ShortTermMomentum::default();

        assert_eq!(factor.name(), "short_term_momentum");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 21);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

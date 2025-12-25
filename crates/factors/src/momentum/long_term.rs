//! Long-term momentum factor (12-month lookback with 1-month skip).

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for long-term momentum factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LongTermMomentumConfig {
    /// Number of trading days to look back (default: 252)
    pub lookback: usize,
    /// Number of recent days to skip to avoid reversal effects (default: 21)
    pub skip_days: usize,
}

impl Default for LongTermMomentumConfig {
    fn default() -> Self {
        Self {
            lookback: 252,
            skip_days: 21,
        }
    }
}

/// Long-term momentum factor measuring 12-month price change with 1-month skip.
///
/// Measures the percentage price change from 12 months ago to 1 month ago:
/// `(P_{t-skip} / P_{t-lookback-skip}) - 1`
///
/// where:
/// - `P_{t-skip}` is the price after skipping recent days
/// - `P_{t-lookback-skip}` is the price from lookback + skip days ago
///
/// The skip period helps avoid short-term reversal effects while capturing
/// long-term trend persistence. This is based on the classic Jegadeesh-Titman
/// momentum strategy (1993).
///
/// Captures long-term trend persistence, useful for:
/// - Classic momentum strategies
/// - Annual rebalancing portfolios
/// - Identifying sustained multi-month trends
/// - Avoiding short-term mean reversion that often occurs in the most recent month
///
/// The skip period is crucial because stocks with very recent strong performance
/// often experience short-term reversals, while those with strong 12-month
/// performance (excluding the most recent month) tend to continue performing well.
#[derive(Debug, Clone, Default)]
pub struct LongTermMomentum {
    config: LongTermMomentumConfig,
}

impl Factor for LongTermMomentum {
    fn name(&self) -> &str {
        "long_term_momentum"
    }

    fn description(&self) -> &str {
        "12-month (252-day) momentum with 1-month skip - long-term trend persistence"
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

impl ConfigurableFactor for LongTermMomentum {
    type Config = LongTermMomentumConfig;

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
    fn test_long_term_momentum_basic() {
        let factor = LongTermMomentum::default();

        // Create test data with 274 days of prices (lookback 252 + skip 21 + 1)
        let dates: Vec<String> = (0..274)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 274];
        // Price goes from 100 to 150 over the full period (50% gain)
        // But we measure from t-273 to t-21
        // At t-273 (index 0): price = 100.0
        // At t-21 (index 252): price = 100.0 + 252 * 0.1984 = 150.0
        // Momentum = (150.0 / 100.0) - 1 = 0.50 (50%)
        let prices: Vec<f64> = (0..274).map(|i| 100.0 + i as f64 * 0.1984).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 10, 1).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that momentum is approximately 50%
        let momentum = result
            .column("long_term_momentum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (momentum - 0.50).abs() < 0.01,
            "Expected ~0.50, got {}",
            momentum
        );
    }

    #[test]
    fn test_long_term_momentum_metadata() {
        let factor = LongTermMomentum::default();

        assert_eq!(factor.name(), "long_term_momentum");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }

    #[test]
    fn test_long_term_momentum_skip_period() {
        let factor = LongTermMomentum::default();

        // Verify that we're using the skip period correctly
        // Create data where the most recent month has a reversal
        let dates: Vec<String> = (0..274)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 274];

        // Price rises from 100 to 150 from t-273 to t-21 (index 0 to 252)
        // Then drops in the most recent month (t-21 to t, index 253 to 273)
        let mut prices: Vec<f64> = (0..253).map(|i| 100.0 + i as f64 * 0.1984).collect();
        // Recent month shows decline
        for i in 253..274 {
            prices.push(150.0 - (i - 252) as f64 * 0.5);
        }

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices.clone(),
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 10, 1).unwrap())
            .unwrap();

        // Momentum should be based on t-21 to t-273, ignoring the recent decline
        let momentum = result
            .column("long_term_momentum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        // Should be around 50% (150/100 - 1), not affected by recent decline
        assert!(
            momentum > 0.4,
            "Momentum should be positive despite recent decline, got {}",
            momentum
        );
    }
}

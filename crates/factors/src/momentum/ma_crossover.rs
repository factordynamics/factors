//! Moving average crossover factor.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for moving average crossover factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MACrossoverConfig {
    /// Short-term moving average window in days (default: 50)
    pub short_window: usize,
    /// Long-term moving average window in days (default: 200)
    pub long_window: usize,
}

impl Default for MACrossoverConfig {
    fn default() -> Self {
        Self {
            short_window: 50,
            long_window: 200,
        }
    }
}

/// Moving average crossover factor - short vs long SMA ratio.
///
/// Measures the relationship between short and long-term moving averages:
/// `SMA_short / SMA_long - 1`
///
/// where:
/// - `SMA_short` is the short-term simple moving average (default: 50-day)
/// - `SMA_long` is the long-term simple moving average (default: 200-day)
///
/// This is a classic technical indicator that captures:
/// - Golden cross: When SMA_short crosses above SMA_long (positive signal)
/// - Death cross: When SMA_short crosses below SMA_long (negative signal)
///
/// Useful for:
/// - Trend identification
/// - Long-term position management
/// - Market timing signals
///
/// Positive values indicate short-term strength relative to long-term average.
/// Negative values indicate short-term weakness relative to long-term average.
#[derive(Debug, Clone, Default)]
pub struct MACrossover {
    config: MACrossoverConfig,
}

impl Factor for MACrossover {
    fn name(&self) -> &str {
        "ma_crossover"
    }

    fn description(&self) -> &str {
        "50-day vs 200-day moving average ratio minus 1"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        self.config.long_window
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

        let short_window = self.config.short_window;
        let long_window = self.config.long_window;

        // Group by symbol and compute MA crossover
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                // Short-term SMA: average of last N prices
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(short_window))
                    .mean()
                    .alias("sma_short"),
                // Long-term SMA: average of last N prices
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(long_window))
                    .mean()
                    .alias("sma_long"),
            ])
            .with_column(((col("sma_short") / col("sma_long")) - lit(1.0)).alias(self.name()))
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for MACrossover {
    type Config = MACrossoverConfig;

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
    fn test_ma_crossover_basic() {
        let factor = MACrossover::default();

        // Create test data with 200 days of prices
        let dates: Vec<String> = (0..200)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 200];
        // Price trending up: recent prices higher than earlier prices
        // This should result in SMA_50 > SMA_200 (positive crossover)
        let prices: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.1).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 7, 18).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that crossover is positive (SMA_50 > SMA_200 for uptrend)
        let crossover = result
            .column("ma_crossover")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            crossover > 0.0,
            "Expected positive crossover for uptrend, got {}",
            crossover
        );
    }

    #[test]
    fn test_ma_crossover_metadata() {
        let factor = MACrossover::default();

        assert_eq!(factor.name(), "ma_crossover");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 200);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

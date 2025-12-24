//! Moving average crossover factor.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Moving average crossover factor - 50-day vs 200-day SMA ratio.
///
/// Measures the relationship between short and long-term moving averages:
/// `SMA_50 / SMA_200 - 1`
///
/// where:
/// - `SMA_50` is the 50-day simple moving average
/// - `SMA_200` is the 200-day simple moving average
///
/// This is a classic technical indicator that captures:
/// - Golden cross: When SMA_50 crosses above SMA_200 (positive signal)
/// - Death cross: When SMA_50 crosses below SMA_200 (negative signal)
///
/// Useful for:
/// - Trend identification
/// - Long-term position management
/// - Market timing signals
///
/// Positive values indicate short-term strength relative to long-term average.
/// Negative values indicate short-term weakness relative to long-term average.
#[derive(Debug, Clone, Default)]
pub struct MACrossover;

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
        200
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

        // Group by symbol and compute MA crossover
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                // 50-day SMA: average of last 50 prices
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(50))
                    .mean()
                    .alias("sma_50"),
                // 200-day SMA: average of last 200 prices
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(200))
                    .mean()
                    .alias("sma_200"),
            ])
            .with_column(((col("sma_50") / col("sma_200")) - lit(1.0)).alias(self.name()))
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
    fn test_ma_crossover_basic() {
        let factor = MACrossover;

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
        let factor = MACrossover;

        assert_eq!(factor.name(), "ma_crossover");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 200);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

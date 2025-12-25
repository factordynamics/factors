//! Short-term momentum factor (1-month lookback).

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Short-term momentum factor measuring 1-month (21 trading days) price change.
///
/// Measures the percentage price change over the past month:
/// `(P_t / P_{t-21}) - 1`
///
/// where:
/// - `P_t` is the current price
/// - `P_{t-21}` is the price 21 trading days ago
///
/// Captures short-term trend persistence, useful for:
/// - High-frequency rebalancing strategies
/// - Identifying recent price breakouts
/// - Combining with longer-term momentum for multi-scale strategies
#[derive(Debug, Clone, Default)]
pub struct ShortTermMomentum;

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
        21
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

        // Group by symbol and compute momentum
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                col("close")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("current_price"),
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit(self.lookback() as i64 + 1)).cast(DataType::Int64),
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

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    #[test]
    fn test_short_term_momentum_basic() {
        let factor = ShortTermMomentum;

        // Create test data with 22 days of prices
        let dates: Vec<String> = (0..22)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 22];
        // Price goes from 100 to 110 (10% gain)
        let prices: Vec<f64> = (0..22).map(|i| 100.0 + i as f64 * 0.476).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 22).unwrap())
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
        let factor = ShortTermMomentum;

        assert_eq!(factor.name(), "short_term_momentum");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 21);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

//! Roll's measure of effective spread.
//!
//! Estimates the bid-ask spread from serial covariance of price changes.
//! This implicit spread measure works without requiring bid-ask quotes.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Roll's measure of effective spread.
///
/// Estimates the effective bid-ask spread using the negative serial covariance
/// of price changes. When prices bounce between bid and ask, consecutive returns
/// have negative covariance proportional to the spread.
///
/// # Interpretation
///
/// - **Higher values**: Wider effective spread, less liquid
/// - **Lower values**: Narrower effective spread, more liquid
///
/// # Computation
///
/// For each security over the lookback period:
/// 1. Calculate price changes: `ΔP_t = close_t - close_{t-1}`
/// 2. Compute covariance: `Cov(ΔP_t, ΔP_{t-1})`
/// 3. Roll spread: `spread = 2 * sqrt(-Cov)` if Cov < 0, else 0
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `close`: Closing price
///
/// # References
///
/// - Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread
///   in an efficient market," Journal of Finance 39(4), 1127-1139.
#[derive(Debug, Clone)]
pub struct RollMeasure {
    lookback: usize,
}

impl RollMeasure {
    /// Creates a new RollMeasure factor with default 20-day lookback.
    pub const fn new() -> Self {
        Self { lookback: 20 }
    }

    /// Creates a RollMeasure factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for RollMeasure {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for RollMeasure {
    fn name(&self) -> &str {
        "roll_spread"
    }

    fn description(&self) -> &str {
        "Implied spread from serial covariance of price changes over 20 days"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        self.lookback
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let result = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())))
            .sort(
                ["symbol", "date"],
                SortMultipleOptions::default().with_order_descending_multi([false, false]),
            )
            // Calculate price changes
            .with_column(
                col("close")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("close_lag1"),
            )
            .with_column((col("close") - col("close_lag1")).alias("price_change"))
            // Calculate lagged price changes
            .with_column(
                col("price_change")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("price_change_lag1"),
            )
            // Calculate rolling covariance manually: Cov(X,Y) = E[XY] - E[X]E[Y]
            .with_column(
                (col("price_change") * col("price_change_lag1")).alias("price_change_product"),
            )
            .with_column(
                col("price_change_product")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("mean_product"),
            )
            .with_column(
                col("price_change")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("mean_price_change"),
            )
            .with_column(
                col("price_change_lag1")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("mean_price_change_lag1"),
            )
            // Covariance = E[XY] - E[X]E[Y]
            .with_column(
                (col("mean_product") - (col("mean_price_change") * col("mean_price_change_lag1")))
                    .alias("price_change_cov"),
            )
            // Roll spread: 2 * sqrt(-cov) if cov < 0, else 0
            .with_column(
                when(col("price_change_cov").lt(lit(0.0)))
                    .then(lit(2.0) * (-col("price_change_cov")).sqrt())
                    .otherwise(lit(0.0))
                    .alias("roll_spread"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("roll_spread")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roll_measure_metadata() {
        let factor = RollMeasure::new();
        assert_eq!(factor.name(), "roll_spread");
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"close"));
    }

    #[test]
    fn test_roll_measure_with_custom_lookback() {
        let factor = RollMeasure::with_lookback(15);
        assert_eq!(factor.lookback(), 15);
    }
}

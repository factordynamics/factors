//! Momentum acceleration factor - rate of change of momentum.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Momentum acceleration factor measuring the difference between short and long-term momentum.
///
/// Measures the acceleration of momentum by comparing:
/// `MOM_3m - MOM_12m`
///
/// where:
/// - `MOM_3m` is 3-month (63-day) momentum
/// - `MOM_12m` is 12-month (252-day) momentum
///
/// This factor captures:
/// - Momentum acceleration: When short-term momentum exceeds long-term
/// - Momentum deceleration: When short-term momentum lags long-term
///
/// Useful for:
/// - Identifying momentum regime changes
/// - Early detection of trend reversals
/// - Short-term tactical allocation
///
/// Positive values indicate accelerating momentum (strengthening trend).
/// Negative values indicate decelerating momentum (weakening trend).
#[derive(Debug, Clone, Default)]
pub struct MomentumAcceleration;

impl Factor for MomentumAcceleration {
    fn name(&self) -> &str {
        "momentum_acceleration"
    }

    fn description(&self) -> &str {
        "Difference between 3-month and 12-month momentum"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        252
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

        // Group by symbol and compute momentum acceleration
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                col("close")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("current_price"),
                // Price 3 months ago (63 days)
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice((lit(0) - lit(64i64)).cast(DataType::Int64), lit(1u32))
                    .first()
                    .alias("price_3m"),
                // Price 12 months ago (252 days)
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice((lit(0) - lit(253i64)).cast(DataType::Int64), lit(1u32))
                    .first()
                    .alias("price_12m"),
            ])
            .with_column(((col("current_price") / col("price_3m")) - lit(1.0)).alias("mom_3m"))
            .with_column(((col("current_price") / col("price_12m")) - lit(1.0)).alias("mom_12m"))
            .with_column((col("mom_3m") - col("mom_12m")).alias(self.name()))
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
    fn test_momentum_acceleration_basic() {
        let factor = MomentumAcceleration;

        // Create test data with 253 days of prices
        let dates: Vec<String> = (0..253)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 253];
        // Price with accelerating growth
        // Exponential growth: recent gains larger than earlier gains
        let prices: Vec<f64> = (0..253)
            .map(|i| {
                100.0 * (1.0 + i as f64 / 1000.0).powi(2) // quadratic growth
            })
            .collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 9, 9).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that acceleration exists (non-null)
        let accel = result
            .column("momentum_acceleration")
            .unwrap()
            .f64()
            .unwrap()
            .get(0);

        assert!(accel.is_some(), "Expected acceleration value");
    }

    #[test]
    fn test_momentum_acceleration_metadata() {
        let factor = MomentumAcceleration;

        assert_eq!(factor.name(), "momentum_acceleration");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

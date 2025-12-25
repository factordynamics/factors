//! Momentum acceleration factor - rate of change of momentum.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for momentum acceleration factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MomentumAccelerationConfig {
    /// Short-term momentum lookback in days (default: 63 = 3 months)
    pub short_term_days: usize,
    /// Long-term momentum lookback in days (default: 252 = 12 months)
    pub long_term_days: usize,
}

impl Default for MomentumAccelerationConfig {
    fn default() -> Self {
        Self {
            short_term_days: 63,
            long_term_days: 252,
        }
    }
}

/// Momentum acceleration factor measuring the difference between short and long-term momentum.
///
/// Measures the acceleration of momentum by comparing:
/// `MOM_short - MOM_long`
///
/// where:
/// - `MOM_short` is short-term momentum (default 63-day / 3-month)
/// - `MOM_long` is long-term momentum (default 252-day / 12-month)
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
pub struct MomentumAcceleration {
    config: MomentumAccelerationConfig,
}

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
        self.config.long_term_days
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

        let short_days = self.config.short_term_days;
        let long_days = self.config.long_term_days;

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
                // Price at short_term_days ago
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit((short_days + 1) as i64)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("price_short"),
                // Price at long_term_days ago
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit((long_days + 1) as i64)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("price_long"),
            ])
            .with_column(
                ((col("current_price") / col("price_short")) - lit(1.0)).alias("mom_short"),
            )
            .with_column(((col("current_price") / col("price_long")) - lit(1.0)).alias("mom_long"))
            .with_column((col("mom_short") - col("mom_long")).alias(self.name()))
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for MomentumAcceleration {
    type Config = MomentumAccelerationConfig;

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
    fn test_momentum_acceleration_basic() {
        let factor = MomentumAcceleration::default();

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
        let factor = MomentumAcceleration::default();

        assert_eq!(factor.name(), "momentum_acceleration");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

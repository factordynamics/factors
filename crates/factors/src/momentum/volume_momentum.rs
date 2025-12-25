//! Volume momentum factor - ratio of short-term to long-term average volume.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Volume momentum factor measuring recent volume relative to baseline.
///
/// Measures the ratio of short-term to long-term average volume:
/// `V_5d / V_20d`
///
/// where:
/// - `V_5d` is the 5-day average volume
/// - `V_20d` is the 20-day average volume
///
/// This factor captures:
/// - Volume surges: When recent volume exceeds typical levels
/// - Volume droughts: When recent volume falls below typical levels
///
/// Useful for:
/// - Confirming price momentum with volume
/// - Identifying unusual trading activity
/// - Liquidity assessment
///
/// Values > 1 indicate increasing volume (accumulation/distribution).
/// Values < 1 indicate decreasing volume (low conviction).
#[derive(Debug, Clone, Default)]
pub struct VolumeMomentum;

impl Factor for VolumeMomentum {
    fn name(&self) -> &str {
        "volume_momentum"
    }

    fn description(&self) -> &str {
        "Ratio of 5-day to 20-day average volume"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "volume"]
    }

    fn lookback(&self) -> usize {
        20
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

        // Group by symbol and compute volume momentum
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                // 5-day average volume
                col("volume")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(5))
                    .mean()
                    .alias("vol_5d"),
                // 20-day average volume
                col("volume")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(20))
                    .mean()
                    .alias("vol_20d"),
            ])
            .with_column(
                when(col("vol_20d").gt(lit(1.0)))
                    .then(col("vol_5d") / col("vol_20d"))
                    .otherwise(col("vol_5d"))
                    .alias(self.name()),
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
    fn test_volume_momentum_basic() {
        let factor = VolumeMomentum;

        // Create test data with 20 days of volume
        let dates: Vec<String> = (0..20)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 20];
        // Volume increasing: first 15 days low volume, last 5 days high volume
        let volumes: Vec<f64> = (0..20)
            .map(|i| {
                if i < 15 {
                    1_000_000.0 // baseline volume
                } else {
                    2_000_000.0 // doubled volume in last 5 days
                }
            })
            .collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "volume" => volumes,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 20).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that volume momentum is > 1 (recent volume higher)
        let vol_mom = result
            .column("volume_momentum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            vol_mom > 1.0,
            "Expected volume momentum > 1, got {}",
            vol_mom
        );

        // Should be approximately 2M / 1.25M = 1.6
        assert!(
            (vol_mom - 1.6).abs() < 0.1,
            "Expected ~1.6, got {}",
            vol_mom
        );
    }

    #[test]
    fn test_volume_momentum_metadata() {
        let factor = VolumeMomentum;

        assert_eq!(factor.name(), "volume_momentum");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "volume"]);
    }
}

//! Days to cover factor.
//!
//! Measures how many days of average trading volume would be needed
//! to cover all short positions. Higher values indicate greater short squeeze risk.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Days to cover factor.
///
/// Computes the number of days of average trading volume needed to cover
/// all outstanding short positions. This measures short squeeze risk and
/// the time horizon for shorts to exit their positions.
///
/// # Interpretation
///
/// - **Higher values**: Longer to cover shorts, higher squeeze risk
/// - **Lower values**: Easier for shorts to cover, lower squeeze risk
///
/// # Computation
///
/// For each security and date:
/// 1. Calculate average daily volume over lookback period
/// 2. Days to cover: `dtc = shares_short / avg_daily_volume`
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `shares_short`: Number of shares sold short
/// - `volume`: Daily trading volume
///
/// # References
///
/// - Dechow, P. M., A. P. Hutton, L. Meulbroek, and R. G. Sloan (2001).
///   "Short-sellers, fundamental analysis, and stock returns," Journal of Financial Economics.
///
/// Configuration for DaysToCover factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaysToCoverConfig {
    /// Number of days to average volume over.
    pub lookback: usize,
    /// Minimum number of periods required for valid calculation.
    pub min_periods: usize,
}

impl Default for DaysToCoverConfig {
    fn default() -> Self {
        Self {
            lookback: 20,
            min_periods: 20,
        }
    }
}

/// Days to cover factor implementation.
#[derive(Debug, Clone)]
pub struct DaysToCover {
    config: DaysToCoverConfig,
}

impl DaysToCover {
    /// Creates a new DaysToCover factor with default 20-day lookback.
    pub const fn new() -> Self {
        Self {
            config: DaysToCoverConfig {
                lookback: 20,
                min_periods: 20,
            },
        }
    }

    /// Creates a DaysToCover factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: DaysToCoverConfig {
                lookback,
                min_periods: lookback,
            },
        }
    }
}

impl Default for DaysToCover {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for DaysToCover {
    fn name(&self) -> &str {
        "days_to_cover"
    }

    fn description(&self) -> &str {
        "Days of average volume needed to cover short positions"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "shares_short", "volume"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback
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
            // Calculate average daily volume over lookback period
            .with_column(
                col("volume")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("avg_volume"),
            )
            // Calculate days to cover: shares_short / avg_volume
            // Add small epsilon to avoid division by zero
            .with_column(
                (col("shares_short") / (col("avg_volume") + lit(1e-10))).alias("days_to_cover"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("days_to_cover")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for DaysToCover {
    type Config = DaysToCoverConfig;

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

    #[test]
    fn test_days_to_cover_metadata() {
        let factor = DaysToCover::new();
        assert_eq!(factor.name(), "days_to_cover");
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"shares_short"));
        assert!(factor.required_columns().contains(&"volume"));
    }

    #[test]
    fn test_days_to_cover_with_custom_lookback() {
        let factor = DaysToCover::with_lookback(10);
        assert_eq!(factor.lookback(), 10);
    }
}

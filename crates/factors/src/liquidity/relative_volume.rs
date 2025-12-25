//! Relative volume factor.
//!
//! Measures current trading volume relative to recent average volume.
//! High relative volume indicates unusual trading activity.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Relative volume factor.
///
/// Computes the ratio of current volume to average volume over a lookback period.
/// This identifies periods of unusual trading activity that may signal information
/// events or liquidity changes.
///
/// # Interpretation
///
/// - **Higher values**: Unusually high volume, increased activity
/// - **Values near 1**: Normal trading volume
/// - **Lower values**: Unusually low volume, decreased activity
///
/// # Computation
///
/// For each security and date:
/// 1. Calculate average volume over lookback period (20 days)
/// 2. Relative volume: `rel_vol = today_volume / avg_volume`
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `volume`: Daily trading volume
///
/// # References
///
/// - Lee, C. M., and M. J. Ready (1991). "Inferring trade direction from intraday data,"
///   Journal of Finance.
///
/// Configuration for RelativeVolume factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeVolumeConfig {
    /// Number of days to average volume over.
    pub lookback: usize,
    /// Minimum number of periods required for valid calculation.
    pub min_periods: usize,
}

impl Default for RelativeVolumeConfig {
    fn default() -> Self {
        Self {
            lookback: 20,
            min_periods: 20,
        }
    }
}

/// Relative volume factor implementation.
#[derive(Debug, Clone)]
pub struct RelativeVolume {
    config: RelativeVolumeConfig,
}

impl RelativeVolume {
    /// Creates a new RelativeVolume factor with default 20-day lookback.
    pub const fn new() -> Self {
        Self {
            config: RelativeVolumeConfig {
                lookback: 20,
                min_periods: 20,
            },
        }
    }

    /// Creates a RelativeVolume factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: RelativeVolumeConfig {
                lookback,
                min_periods: lookback,
            },
        }
    }
}

impl Default for RelativeVolume {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for RelativeVolume {
    fn name(&self) -> &str {
        "relative_volume"
    }

    fn description(&self) -> &str {
        "Current volume relative to 20-day average volume"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "volume"]
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
            // Calculate average volume over lookback period
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
            // Calculate relative volume: today_volume / avg_volume
            // Add small epsilon to avoid division by zero
            .with_column(
                (col("volume") / (col("avg_volume") + lit(1e-10))).alias("relative_volume"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("relative_volume")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for RelativeVolume {
    type Config = RelativeVolumeConfig;

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
    fn test_relative_volume_metadata() {
        let factor = RelativeVolume::new();
        assert_eq!(factor.name(), "relative_volume");
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"volume"));
    }

    #[test]
    fn test_relative_volume_with_custom_lookback() {
        let factor = RelativeVolume::with_lookback(30);
        assert_eq!(factor.lookback(), 30);
    }
}

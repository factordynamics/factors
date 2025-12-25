//! Relative volume factor.
//!
//! Measures current trading volume relative to recent average volume.
//! High relative volume indicates unusual trading activity.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

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
#[derive(Debug, Clone)]
pub struct RelativeVolume {
    lookback: usize,
}

impl RelativeVolume {
    /// Creates a new RelativeVolume factor with default 20-day lookback.
    pub const fn new() -> Self {
        Self { lookback: 20 }
    }

    /// Creates a RelativeVolume factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
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
            // Calculate average volume over lookback period
            .with_column(
                col("volume")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
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

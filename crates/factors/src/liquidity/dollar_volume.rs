//! Dollar volume factor.
//!
//! Measures liquidity through the dollar value of shares traded.
//! Higher dollar volume indicates more liquid securities with lower transaction costs.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Dollar volume factor.
///
/// Computes the average dollar volume (Price × Volume) over a lookback period.
/// This measures market liquidity by quantifying the actual dollar value traded.
///
/// # Interpretation
///
/// - **Higher values**: More liquid, larger dollar amounts traded
/// - **Lower values**: Less liquid, smaller dollar amounts traded
///
/// # Computation
///
/// For each security and date:
/// 1. Calculate daily dollar volume: `dollar_volume_t = close_t * volume_t`
/// 2. Average over the lookback period (20 days)
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `close`: Closing price
/// - `volume`: Daily trading volume
///
/// # References
///
/// - Brennan, M. J., and A. Subrahmanyam (1996). "Market microstructure and asset pricing:
///   On the compensation for illiquidity in stock returns," Journal of Financial Economics.
#[derive(Debug, Clone)]
pub struct DollarVolume {
    lookback: usize,
}

impl DollarVolume {
    /// Creates a new DollarVolume factor with default 20-day lookback.
    pub const fn new() -> Self {
        Self { lookback: 20 }
    }

    /// Creates a DollarVolume factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for DollarVolume {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for DollarVolume {
    fn name(&self) -> &str {
        "dollar_volume"
    }

    fn description(&self) -> &str {
        "Average dollar volume (price * volume) over 20 days"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close", "volume"]
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
            // Calculate daily dollar volume
            .with_column((col("close") * col("volume")).alias("daily_dollar_volume"))
            // Rolling mean over lookback period
            .with_column(
                col("daily_dollar_volume")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("dollar_volume"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("dollar_volume")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dollar_volume_metadata() {
        let factor = DollarVolume::new();
        assert_eq!(factor.name(), "dollar_volume");
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"close"));
        assert!(factor.required_columns().contains(&"volume"));
    }

    #[test]
    fn test_dollar_volume_with_custom_lookback() {
        let factor = DollarVolume::with_lookback(30);
        assert_eq!(factor.lookback(), 30);
    }
}

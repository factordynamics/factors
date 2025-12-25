//! Maximum drawdown factor - peak-to-trough decline measure.
//!
//! Maximum drawdown measures the largest peak-to-trough decline in price
//! over a lookback period. This captures the worst-case loss an investor
//! would have experienced.
//!
//! Formula: `MaxDD = max((peak - trough) / peak)` over lookback period
//!
//! Higher (more negative) drawdown indicates greater historical downside risk.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Maximum drawdown factor.
///
/// Computes the maximum peak-to-trough decline as a percentage over a
/// lookback period. This is a measure of downside risk and capital preservation.
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `max_drawdown`
#[derive(Debug, Clone)]
pub struct MaxDrawdown {
    lookback: usize,
}

impl MaxDrawdown {
    /// Create a new MaxDrawdown factor with default lookback (252 days).
    pub const fn new() -> Self {
        Self { lookback: 252 }
    }

    /// Create a MaxDrawdown factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for MaxDrawdown {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for MaxDrawdown {
    fn name(&self) -> &str {
        "max_drawdown"
    }

    fn description(&self) -> &str {
        "Maximum peak-to-trough decline - worst historical loss over lookback period"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Volatility
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
        // Filter to dates up to and including the target date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())));

        // Sort by symbol and date
        let sorted = filtered.sort(
            ["symbol", "date"],
            SortMultipleOptions::default().with_order_descending_multi([false, false]),
        );

        // Compute rolling maximum (peak) price
        let with_peak = sorted.with_column(
            col("close")
                .rolling_max(RollingOptionsFixedWindow {
                    window_size: self.lookback,
                    min_periods: self.lookback,
                    ..Default::default()
                })
                .over([col("symbol")])
                .alias("rolling_peak"),
        );

        // Compute drawdown: (current price - peak) / peak
        // Then take the minimum (most negative) drawdown
        let result = with_peak
            .with_column(
                ((col("close") - col("rolling_peak")) / col("rolling_peak")).alias("drawdown"),
            )
            .with_column(
                col("drawdown")
                    .rolling_min(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("max_drawdown"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("max_drawdown")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown_lookback() {
        let factor = MaxDrawdown::with_lookback(126);
        assert_eq!(factor.lookback(), 126);
    }

    #[test]
    fn test_max_drawdown_metadata() {
        let factor = MaxDrawdown::new();
        assert_eq!(factor.name(), "max_drawdown");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

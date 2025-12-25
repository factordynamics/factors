//! Return kurtosis factor - tail heaviness measure.
//!
//! Kurtosis measures the "tailedness" of the return distribution. High kurtosis
//! indicates fat tails (more extreme returns than a normal distribution), while
//! low kurtosis indicates thin tails. We compute excess kurtosis (kurtosis - 3).
//!
//! Formula: `Excess Kurtosis = E[(R - μ)⁴] / σ⁴ - 3` (fourth standardized moment minus 3)
//!
//! Positive excess kurtosis indicates fat tails and greater risk of extreme moves.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Return kurtosis factor.
///
/// Computes the excess kurtosis (fourth standardized moment minus 3) of daily
/// returns over a lookback period. This measures the heaviness of distribution tails.
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `return_kurtosis`
#[derive(Debug, Clone)]
pub struct ReturnKurtosis {
    lookback: usize,
}

impl ReturnKurtosis {
    /// Create a new ReturnKurtosis factor with default lookback (252 days).
    pub const fn new() -> Self {
        Self { lookback: 252 }
    }

    /// Create a ReturnKurtosis factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for ReturnKurtosis {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for ReturnKurtosis {
    fn name(&self) -> &str {
        "return_kurtosis"
    }

    fn description(&self) -> &str {
        "Return distribution tail heaviness - excess kurtosis (fourth standardized moment minus 3)"
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

        // Sort and compute returns using shift
        let with_returns = filtered
            .sort(
                ["symbol", "date"],
                SortMultipleOptions::default().with_order_descending_multi([false, false]),
            )
            .with_column(
                col("close")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("close_lag"),
            )
            .with_column(((col("close") - col("close_lag")) / col("close_lag")).alias("return"));

        // Compute mean and std for standardization
        let with_stats = with_returns
            .with_column(
                col("return")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("return_mean"),
            )
            .with_column(
                col("return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("return_std"),
            );

        // Compute excess kurtosis: E[(x - μ)⁴] / σ⁴ - 3
        // First compute standardized returns, then raise to 4th power, then take mean, then subtract 3
        let result = with_stats
            .with_column(
                ((col("return") - col("return_mean")) / col("return_std")).alias("z_score"),
            )
            .with_column(col("z_score").pow(4.0).alias("z_fourth"))
            .with_column(
                col("z_fourth")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("raw_kurtosis"),
            )
            // Subtract 3 for excess kurtosis
            .with_column((col("raw_kurtosis") - lit(3.0)).alias("return_kurtosis"))
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("return_kurtosis")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_return_kurtosis_lookback() {
        let factor = ReturnKurtosis::with_lookback(126);
        assert_eq!(factor.lookback(), 126);
    }

    #[test]
    fn test_return_kurtosis_metadata() {
        let factor = ReturnKurtosis::new();
        assert_eq!(factor.name(), "return_kurtosis");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

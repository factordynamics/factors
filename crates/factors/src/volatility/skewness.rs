//! Return skewness factor - asymmetry measure.
//!
//! Skewness measures the asymmetry of the return distribution. Negative skewness
//! indicates a distribution with a longer left tail (more extreme negative returns),
//! while positive skewness indicates a longer right tail.
//!
//! Formula: `Skewness = E[(R - μ)³] / σ³` (third standardized moment)
//!
//! Negative skewness is associated with crash risk and downside asymmetry.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for the ReturnSkewness factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReturnSkewnessConfig {
    /// Number of trading days for the rolling calculation.
    pub lookback: usize,
    /// Minimum number of periods required for a valid calculation.
    pub min_periods: usize,
}

impl Default for ReturnSkewnessConfig {
    fn default() -> Self {
        Self {
            lookback: 252,
            min_periods: 252,
        }
    }
}

/// Return skewness factor.
///
/// Computes the third standardized moment of daily returns over a lookback period.
/// This measures the asymmetry of the return distribution.
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `return_skewness`
#[derive(Debug, Clone)]
pub struct ReturnSkewness {
    config: ReturnSkewnessConfig,
}

impl ReturnSkewness {
    /// Create a new ReturnSkewness factor with default lookback (252 days).
    pub fn new() -> Self {
        Self {
            config: ReturnSkewnessConfig::default(),
        }
    }

    /// Create a ReturnSkewness factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: ReturnSkewnessConfig {
                lookback,
                min_periods: lookback,
            },
        }
    }
}

impl Default for ReturnSkewness {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigurableFactor for ReturnSkewness {
    type Config = ReturnSkewnessConfig;

    fn with_config(config: Self::Config) -> Self {
        Self { config }
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Factor for ReturnSkewness {
    fn name(&self) -> &str {
        "return_skewness"
    }

    fn description(&self) -> &str {
        "Return distribution asymmetry - third standardized moment of daily returns"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Volatility
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback
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
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("return_mean"),
            )
            .with_column(
                col("return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("return_std"),
            );

        // Compute skewness: E[(x - μ)³] / σ³
        // First compute standardized returns, then cube them, then take the mean
        let result = with_stats
            .with_column(
                ((col("return") - col("return_mean")) / col("return_std")).alias("z_score"),
            )
            .with_column(col("z_score").pow(3.0).alias("z_cubed"))
            .with_column(
                col("z_cubed")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("return_skewness"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("return_skewness")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_return_skewness_lookback() {
        let factor = ReturnSkewness::with_lookback(126);
        assert_eq!(factor.lookback(), 126);
    }

    #[test]
    fn test_return_skewness_metadata() {
        let factor = ReturnSkewness::new();
        assert_eq!(factor.name(), "return_skewness");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

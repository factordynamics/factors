//! Short interest ratio factor.
//!
//! Measures the level of short selling activity relative to float.
//! High short interest may indicate bearish sentiment or anticipated price declines.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Short interest ratio factor.
///
/// Computes the ratio of shares sold short to the float (shares available for trading).
/// This measures the intensity of short selling and potential for short squeezes.
///
/// # Interpretation
///
/// - **Higher values**: More short interest, potentially bearish sentiment
/// - **Lower values**: Less short interest, less bearish pressure
///
/// # Computation
///
/// For each security and date:
/// 1. Calculate short interest ratio: `short_ratio = shares_short / float_shares`
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `shares_short`: Number of shares sold short
/// - `float_shares`: Shares available for trading (float)
///
/// # References
///
/// - Dechow, P. M., A. P. Hutton, L. Meulbroek, and R. G. Sloan (2001).
///   "Short-sellers, fundamental analysis, and stock returns," Journal of Financial Economics.
///
/// Configuration for ShortInterestRatio factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortInterestRatioConfig {
    /// Lookback period (typically 1 for current ratio).
    pub lookback: usize,
}

impl Default for ShortInterestRatioConfig {
    fn default() -> Self {
        Self { lookback: 1 }
    }
}

/// Short interest ratio factor implementation.
#[derive(Debug, Clone)]
pub struct ShortInterestRatio {
    config: ShortInterestRatioConfig,
}

impl ShortInterestRatio {
    /// Creates a new ShortInterestRatio factor with default 1-day lookback.
    pub const fn new() -> Self {
        Self {
            config: ShortInterestRatioConfig { lookback: 1 },
        }
    }

    /// Creates a ShortInterestRatio factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: ShortInterestRatioConfig { lookback },
        }
    }
}

impl Default for ShortInterestRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for ShortInterestRatio {
    fn name(&self) -> &str {
        "short_interest_ratio"
    }

    fn description(&self) -> &str {
        "Ratio of shares sold short to float"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "shares_short", "float_shares"]
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
            // Calculate short interest ratio
            // Add small epsilon to avoid division by zero
            .with_column(
                (col("shares_short") / (col("float_shares") + lit(1e-10)))
                    .alias("short_interest_ratio"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("short_interest_ratio")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for ShortInterestRatio {
    type Config = ShortInterestRatioConfig;

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
    fn test_short_interest_ratio_metadata() {
        let factor = ShortInterestRatio::new();
        assert_eq!(factor.name(), "short_interest_ratio");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"shares_short"));
        assert!(factor.required_columns().contains(&"float_shares"));
    }

    #[test]
    fn test_short_interest_ratio_with_custom_lookback() {
        let factor = ShortInterestRatio::with_lookback(5);
        assert_eq!(factor.lookback(), 5);
    }
}

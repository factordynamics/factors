//! Amihud illiquidity factor.
//!
//! Measures price impact of trading through the relationship between
//! absolute returns and dollar volume.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Amihud illiquidity factor.
///
/// Computes the average ratio of absolute return to dollar volume over a
/// 21-day lookback period. This measures the price impact per dollar of
/// trading volume, serving as a proxy for illiquidity and transaction costs.
///
/// # Interpretation
///
/// - **Higher values**: More illiquid, larger price impact per dollar traded
/// - **Lower values**: More liquid, smaller price impact per dollar traded
///
/// Note: This factor measures *illiquidity* - higher scores indicate less liquid securities.
///
/// # Computation
///
/// For each security and date:
/// 1. Calculate daily return: `return_t = (close_t - close_{t-1}) / close_{t-1}`
/// 2. Calculate dollar volume: `dollar_volume_t = close_t * volume_t`
/// 3. Calculate daily illiquidity: `illiq_t = |return_t| / dollar_volume_t`
/// 4. Average over the lookback period (21 days)
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
/// - Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and
///   time-series effects," Journal of Financial Markets 5, 31-56.
///
/// Configuration for AmihudIlliquidity factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmihudIlliquidityConfig {
    /// Number of days to average illiquidity over.
    pub lookback: usize,
    /// Minimum number of periods required for valid calculation.
    pub min_periods: usize,
}

impl Default for AmihudIlliquidityConfig {
    fn default() -> Self {
        Self {
            lookback: 21,
            min_periods: 21,
        }
    }
}

/// Amihud illiquidity factor implementation.
#[derive(Debug, Clone)]
pub struct AmihudIlliquidity {
    config: AmihudIlliquidityConfig,
}

impl AmihudIlliquidity {
    /// Creates a new AmihudIlliquidity factor with default 21-day lookback.
    pub const fn new() -> Self {
        Self {
            config: AmihudIlliquidityConfig {
                lookback: 21,
                min_periods: 21,
            },
        }
    }

    /// Creates an AmihudIlliquidity factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: AmihudIlliquidityConfig {
                lookback,
                min_periods: lookback,
            },
        }
    }
}

impl Default for AmihudIlliquidity {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for AmihudIlliquidity {
    fn name(&self) -> &str {
        "amihud_illiquidity"
    }

    fn description(&self) -> &str {
        "Average ratio of absolute return to dollar volume over 21 days"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close", "volume"]
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
            // Calculate lagged close price
            .with_column(
                col("close")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("close_lag"),
            )
            // Calculate daily return
            .with_column(
                ((col("close") - col("close_lag")) / col("close_lag")).alias("daily_return"),
            )
            // Calculate dollar volume (close * volume)
            .with_column((col("close") * col("volume")).alias("dollar_volume"))
            // Calculate daily Amihud illiquidity: |return| / dollar_volume
            // Add small epsilon to avoid division by zero
            .with_column(
                (col("daily_return").abs() / (col("dollar_volume") + lit(1e-10)))
                    .alias("daily_illiquidity"),
            )
            // Rolling mean over lookback period
            .with_column(
                col("daily_illiquidity")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("amihud_illiquidity"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("amihud_illiquidity")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for AmihudIlliquidity {
    type Config = AmihudIlliquidityConfig;

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
    fn test_amihud_illiquidity_metadata() {
        let factor = AmihudIlliquidity::new();
        assert_eq!(factor.name(), "amihud_illiquidity");
        assert_eq!(factor.lookback(), 21);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"close"));
        assert!(factor.required_columns().contains(&"volume"));
    }

    #[test]
    fn test_amihud_with_custom_lookback() {
        let factor = AmihudIlliquidity::with_lookback(10);
        assert_eq!(factor.lookback(), 10);
    }
}

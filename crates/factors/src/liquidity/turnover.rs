//! Turnover ratio factor.
//!
//! Measures liquidity through trading volume relative to shares outstanding.
//! Higher turnover indicates more liquid securities with lower transaction costs.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Turnover ratio factor.
///
/// Computes the average turnover ratio (Volume / Shares Outstanding) over a
/// 21-day lookback period. This measures liquidity by comparing daily trading
/// volume to the total shares available for trading.
///
/// # Interpretation
///
/// - **Higher values**: More liquid, easier to trade without price impact
/// - **Lower values**: Less liquid, higher transaction costs
///
/// # Computation
///
/// For each security and date:
/// 1. Calculate daily turnover: `turnover_t = volume_t / shares_outstanding_t`
/// 2. Average over the lookback period (21 days)
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `volume`: Daily trading volume
/// - `shares_outstanding`: Total shares outstanding
///
/// # References
///
/// - Datar, V. T., Y. Naik, and R. Radcliffe (1998). "Liquidity and stock returns:
///   An alternative test," Journal of Financial Markets.
///
/// Configuration for TurnoverRatio factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnoverRatioConfig {
    /// Number of days to average turnover over.
    pub lookback: usize,
    /// Minimum number of periods required for valid calculation.
    pub min_periods: usize,
}

impl Default for TurnoverRatioConfig {
    fn default() -> Self {
        Self {
            lookback: 21,
            min_periods: 21,
        }
    }
}

/// Turnover ratio factor implementation.
#[derive(Debug, Clone)]
pub struct TurnoverRatio {
    config: TurnoverRatioConfig,
}

impl TurnoverRatio {
    /// Creates a new TurnoverRatio factor with default 21-day lookback.
    pub const fn new() -> Self {
        Self {
            config: TurnoverRatioConfig {
                lookback: 21,
                min_periods: 21,
            },
        }
    }

    /// Creates a TurnoverRatio factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: TurnoverRatioConfig {
                lookback,
                min_periods: lookback,
            },
        }
    }
}

impl Default for TurnoverRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for TurnoverRatio {
    fn name(&self) -> &str {
        "turnover_ratio"
    }

    fn description(&self) -> &str {
        "Average trading volume as a fraction of shares outstanding over 21 days"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "volume", "shares_outstanding"]
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
            // Calculate daily turnover ratio
            .with_column((col("volume") / col("shares_outstanding")).alias("daily_turnover"))
            // Rolling mean over lookback period
            .with_column(
                col("daily_turnover")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("turnover_ratio"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("turnover_ratio")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for TurnoverRatio {
    type Config = TurnoverRatioConfig;

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
    fn test_turnover_ratio_metadata() {
        let factor = TurnoverRatio::new();
        assert_eq!(factor.name(), "turnover_ratio");
        assert_eq!(factor.lookback(), 21);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"volume"));
        assert!(factor.required_columns().contains(&"shares_outstanding"));
    }
}

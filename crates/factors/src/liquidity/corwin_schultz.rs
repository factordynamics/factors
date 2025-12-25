//! Corwin-Schultz high-low spread estimator.
//!
//! Estimates the bid-ask spread using daily high and low prices.
//! This measure works without requiring intraday data or bid-ask quotes.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Corwin-Schultz high-low spread estimator.
///
/// Estimates the effective bid-ask spread using the relationship between
/// daily high-low ranges and two-day high-low ranges. The intuition is that
/// the daily range reflects both volatility and the bid-ask spread.
///
/// # Interpretation
///
/// - **Higher values**: Wider effective spread, less liquid
/// - **Lower values**: Narrower effective spread, more liquid
///
/// # Computation
///
/// For each security over the lookback period:
/// 1. Calculate single-period high-low ratio: `β = (ln(H_t/L_t))^2 + (ln(H_{t-1}/L_{t-1}))^2`
/// 2. Calculate two-period high-low ratio: `γ = (ln(max(H_t, H_{t-1})/min(L_t, L_{t-1})))^2`
/// 3. Estimate spread component: `α = (sqrt(2β) - sqrt(β)) / (3 - 2*sqrt(2)) - sqrt(γ/(3-2*sqrt(2)))`
/// 4. Spread estimate: `S = 2(e^α - 1) / (1 + e^α)`
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `high`: Daily high price
/// - `low`: Daily low price
///
/// # References
///
/// - Corwin, S. A., and P. Schultz (2012). "A simple way to estimate bid-ask spreads
///   from daily high and low prices," Journal of Finance 67(2), 719-760.
///
/// Configuration for CorwinSchultz factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorwinSchultzConfig {
    /// Number of days to average spread estimate over.
    pub lookback: usize,
    /// Minimum number of periods required for valid calculation.
    pub min_periods: usize,
}

impl Default for CorwinSchultzConfig {
    fn default() -> Self {
        Self {
            lookback: 20,
            min_periods: 20,
        }
    }
}

/// Corwin-Schultz spread estimator implementation.
#[derive(Debug, Clone)]
pub struct CorwinSchultz {
    config: CorwinSchultzConfig,
}

impl CorwinSchultz {
    /// Creates a new CorwinSchultz factor with default 20-day lookback.
    pub const fn new() -> Self {
        Self {
            config: CorwinSchultzConfig {
                lookback: 20,
                min_periods: 20,
            },
        }
    }

    /// Creates a CorwinSchultz factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: CorwinSchultzConfig {
                lookback,
                min_periods: lookback,
            },
        }
    }
}

impl Default for CorwinSchultz {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for CorwinSchultz {
    fn name(&self) -> &str {
        "corwin_schultz_spread"
    }

    fn description(&self) -> &str {
        "High-low spread estimator using 2-day high/low ranges over 20 days"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "high", "low"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let sqrt2 = std::f64::consts::SQRT_2;
        let denominator = 3.0 - 2.0 * sqrt2;

        let result = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())))
            .sort(
                ["symbol", "date"],
                SortMultipleOptions::default().with_order_descending_multi([false, false]),
            )
            // Add small epsilon to avoid log(0)
            .with_column((col("high") + lit(1e-10)).alias("high_adj"))
            .with_column((col("low") + lit(1e-10)).alias("low_adj"))
            // Calculate single-period high-low squared
            .with_column(
                (col("high_adj") / col("low_adj"))
                    .log(std::f64::consts::E)
                    .pow(lit(2.0))
                    .alias("hl_squared"),
            )
            // Get lagged values for 2-day calculations
            .with_column(
                col("high_adj")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("high_lag1"),
            )
            .with_column(
                col("low_adj")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("low_lag1"),
            )
            .with_column(
                col("hl_squared")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("hl_squared_lag1"),
            )
            // Beta: sum of two single-period squared high-low ratios
            .with_column((col("hl_squared") + col("hl_squared_lag1")).alias("beta"))
            // Gamma: two-period high-low ratio squared
            // Use max/min functions for two-period high/low
            .with_column(
                when(col("high_adj").gt(col("high_lag1")))
                    .then(col("high_adj"))
                    .otherwise(col("high_lag1"))
                    .alias("two_period_high"),
            )
            .with_column(
                when(col("low_adj").lt(col("low_lag1")))
                    .then(col("low_adj"))
                    .otherwise(col("low_lag1"))
                    .alias("two_period_low"),
            )
            .with_column(
                (col("two_period_high") / col("two_period_low"))
                    .log(std::f64::consts::E)
                    .pow(lit(2.0))
                    .alias("gamma"),
            )
            // Alpha: spread component
            // α = (sqrt(2β) - sqrt(β)) / (3 - 2*sqrt(2)) - sqrt(γ/(3-2*sqrt(2)))
            .with_column(
                ((lit(sqrt2) * col("beta").sqrt() - col("beta").sqrt()) / lit(denominator)
                    - (col("gamma") / lit(denominator)).sqrt())
                .alias("alpha"),
            )
            // Spread: S = 2(e^α - 1) / (1 + e^α)
            .with_column(
                (lit(2.0) * (col("alpha").exp() - lit(1.0)) / (lit(1.0) + col("alpha").exp()))
                    .alias("daily_spread"),
            )
            // Handle negative spreads (set to 0)
            .with_column(
                when(col("daily_spread").lt(lit(0.0)))
                    .then(lit(0.0))
                    .otherwise(col("daily_spread"))
                    .alias("daily_spread"),
            )
            // Rolling mean over lookback period
            .with_column(
                col("daily_spread")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("corwin_schultz_spread"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("corwin_schultz_spread")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for CorwinSchultz {
    type Config = CorwinSchultzConfig;

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
    fn test_corwin_schultz_metadata() {
        let factor = CorwinSchultz::new();
        assert_eq!(factor.name(), "corwin_schultz_spread");
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"high"));
        assert!(factor.required_columns().contains(&"low"));
    }

    #[test]
    fn test_corwin_schultz_with_custom_lookback() {
        let factor = CorwinSchultz::with_lookback(30);
        assert_eq!(factor.lookback(), 30);
    }
}

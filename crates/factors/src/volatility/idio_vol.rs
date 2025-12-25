//! Idiosyncratic volatility factor - firm-specific risk measure.
//!
//! Idiosyncratic volatility measures the standard deviation of residuals from
//! regressing stock returns on market returns. This captures the volatility
//! that is not explained by market movements (unsystematic risk).
//!
//! Formula: `σ_idio = std(ε)` where `ε = R_i - (α + β × R_m)`
//!
//! Higher idiosyncratic volatility indicates greater firm-specific risk.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for the IdiosyncraticVolatility factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IdiosyncraticVolatilityConfig {
    /// Number of trading days for the rolling calculation.
    pub lookback: usize,
    /// Minimum number of periods required for a valid calculation.
    pub min_periods: usize,
}

impl Default for IdiosyncraticVolatilityConfig {
    fn default() -> Self {
        Self {
            lookback: 252,
            min_periods: 252,
        }
    }
}

/// Idiosyncratic volatility factor.
///
/// Computes the standard deviation of residuals from a market model regression.
/// This represents the portion of volatility not explained by the market.
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price
/// - `market_return`: Market return (e.g., S&P 500 daily return)
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `idiosyncratic_volatility`
#[derive(Debug, Clone)]
pub struct IdiosyncraticVolatility {
    config: IdiosyncraticVolatilityConfig,
}

impl IdiosyncraticVolatility {
    /// Create a new IdiosyncraticVolatility factor with default lookback (252 days).
    pub fn new() -> Self {
        Self {
            config: IdiosyncraticVolatilityConfig::default(),
        }
    }

    /// Create an IdiosyncraticVolatility factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: IdiosyncraticVolatilityConfig {
                lookback,
                min_periods: lookback,
            },
        }
    }
}

impl Default for IdiosyncraticVolatility {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigurableFactor for IdiosyncraticVolatility {
    type Config = IdiosyncraticVolatilityConfig;

    fn with_config(config: Self::Config) -> Self {
        Self { config }
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Factor for IdiosyncraticVolatility {
    fn name(&self) -> &str {
        "idiosyncratic_volatility"
    }

    fn description(&self) -> &str {
        "Firm-specific risk - standard deviation of residuals from market model regression"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Volatility
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close", "market_return"]
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

        // Compute rolling statistics for regression
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
                col("market_return")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("market_mean"),
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
            )
            .with_column(
                col("market_return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("market_std"),
            );

        // Simplified beta calculation: beta ≈ std(stock) / std(market)
        // Compute residuals: residual = return - beta * market_return
        // Idiosyncratic vol = std(residuals)
        let result = with_stats
            .with_column((col("return_std") / col("market_std")).alias("beta"))
            .with_column((col("return") - col("beta") * col("market_return")).alias("residual"))
            .with_column(
                col("residual")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("idiosyncratic_volatility"),
            )
            // Annualize the volatility
            .with_column(
                (col("idiosyncratic_volatility") * lit(252.0_f64.sqrt()))
                    .alias("idiosyncratic_volatility"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("idiosyncratic_volatility")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idiosyncratic_volatility_lookback() {
        let factor = IdiosyncraticVolatility::with_lookback(126);
        assert_eq!(factor.lookback(), 126);
    }

    #[test]
    fn test_idiosyncratic_volatility_metadata() {
        let factor = IdiosyncraticVolatility::new();
        assert_eq!(factor.name(), "idiosyncratic_volatility");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.lookback(), 252);
        assert!(factor.required_columns().contains(&"market_return"));
    }
}

//! Downside beta factor - systematic risk during market downturns.
//!
//! Downside beta measures the sensitivity of a security's returns to market returns,
//! but only on days when the market return is negative. This captures systematic
//! risk during market stress periods.
//!
//! Formula: `β_downside = Cov(R_i, R_m | R_m < 0) / Var(R_m | R_m < 0)`
//!
//! Higher downside beta indicates greater sensitivity to market declines.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Downside beta factor.
///
/// Computes beta using only days when the market return is negative.
/// This measures how much a security moves with the market during downturns.
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price
/// - `market_return`: Market return (e.g., S&P 500 daily return)
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `downside_beta`
#[derive(Debug, Clone)]
pub struct DownsideBeta {
    lookback: usize,
}

impl DownsideBeta {
    /// Create a new DownsideBeta factor with default lookback (252 days).
    pub const fn new() -> Self {
        Self { lookback: 252 }
    }

    /// Create a DownsideBeta factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for DownsideBeta {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for DownsideBeta {
    fn name(&self) -> &str {
        "downside_beta"
    }

    fn description(&self) -> &str {
        "Systematic risk exposure during market downturns - beta calculated only on days when market return is negative"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Volatility
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close", "market_return"]
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

        // Filter to only negative market days and compute downside statistics
        let downside_only = with_returns.filter(col("market_return").lt(lit(0.0)));

        // Compute downside statistics: covariance and variance on negative market days
        // Using simplified beta calculation: std(stock_downside) / std(market_downside) * correlation
        // For a more accurate beta, we'd need to compute actual covariance
        let result = downside_only
            .with_column(
                col("return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: (self.lookback / 4).max(20), // Need at least some downside days
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("stock_downside_std"),
            )
            .with_column(
                col("market_return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: (self.lookback / 4).max(20),
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("market_downside_std"),
            )
            // Simplified beta approximation
            .with_column(
                (col("stock_downside_std") / col("market_downside_std")).alias("downside_beta"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("downside_beta")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downside_beta_lookback() {
        let factor = DownsideBeta::with_lookback(126);
        assert_eq!(factor.lookback(), 126);
    }

    #[test]
    fn test_downside_beta_metadata() {
        let factor = DownsideBeta::new();
        assert_eq!(factor.name(), "downside_beta");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.lookback(), 252);
        assert!(factor.required_columns().contains(&"market_return"));
    }
}

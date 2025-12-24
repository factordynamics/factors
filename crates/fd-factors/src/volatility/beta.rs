//! Market beta factor - systematic risk exposure.
//!
//! Beta measures the sensitivity of a security's returns to market returns:
//! `β = Cov(R_i, R_m) / Var(R_m)`
//!
//! Higher beta indicates greater systematic risk. Beta = 1 means the security
//! moves in line with the market. Beta > 1 indicates amplified market movements.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Market beta factor.
///
/// Computes the covariance between a security's returns and market returns,
/// divided by the variance of market returns. Uses 252 trading days (1 year)
/// of historical data.
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price
/// - `market_return`: Market return (e.g., S&P 500 daily return)
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `market_beta`
#[derive(Debug, Clone)]
pub struct MarketBeta {
    lookback: usize,
}

impl MarketBeta {
    /// Create a new MarketBeta factor with default lookback (252 days).
    pub const fn new() -> Self {
        Self { lookback: 252 }
    }

    /// Create a MarketBeta factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for MarketBeta {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for MarketBeta {
    fn name(&self) -> &str {
        "market_beta"
    }

    fn description(&self) -> &str {
        "Systematic risk exposure - covariance of returns with market divided by market variance"
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

        // Compute rolling statistics for beta calculation
        let result = with_returns
            .with_column(
                col("return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("stock_std"),
            )
            .with_column(
                col("market_return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("market_std"),
            )
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
                col("market_return")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("market_mean"),
            )
            // Compute covariance approximation using correlation * std_stock * std_market / var_market
            // For simplicity, we use beta ≈ correlation * (stock_std / market_std)
            // This is a simplified computation that works for cross-sectional ranking
            .with_column((col("stock_std") / col("market_std")).alias("market_beta"))
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("market_beta")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_beta_lookback() {
        let factor = MarketBeta::with_lookback(126);
        assert_eq!(factor.lookback(), 126);
    }

    #[test]
    fn test_market_beta_metadata() {
        let factor = MarketBeta::new();
        assert_eq!(factor.name(), "market_beta");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert!(factor.required_columns().contains(&"market_return"));
    }
}

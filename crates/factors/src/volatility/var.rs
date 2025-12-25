//! Tail risk (Value at Risk) factor - downside risk measure.
//!
//! Value at Risk (VaR) measures the maximum expected loss at a given confidence level.
//! We use the historical VaR method with a 5% confidence level (95% VaR), which is
//! the 5th percentile of the return distribution.
//!
//! Formula: `VaR_5% = 5th percentile of daily returns`
//!
//! More negative VaR indicates greater tail risk (larger potential losses).

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Value at Risk (VaR) factor.
///
/// Computes the 5th percentile of daily returns over a lookback period.
/// This represents the threshold below which 5% of returns fall, capturing
/// tail risk and downside exposure.
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `value_at_risk`
#[derive(Debug, Clone)]
pub struct ValueAtRisk {
    lookback: usize,
}

impl ValueAtRisk {
    /// Create a new ValueAtRisk factor with default lookback (252 days).
    pub const fn new() -> Self {
        Self { lookback: 252 }
    }

    /// Create a ValueAtRisk factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for ValueAtRisk {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for ValueAtRisk {
    fn name(&self) -> &str {
        "value_at_risk"
    }

    fn description(&self) -> &str {
        "Tail risk measure - 5th percentile of daily returns (historical VaR)"
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

        // Compute rolling 5th percentile (VaR)
        let result = with_returns
            .with_column(
                col("return")
                    .rolling_quantile(
                        QuantileMethod::Linear,
                        0.05, // 5th percentile
                        RollingOptionsFixedWindow {
                            window_size: self.lookback,
                            min_periods: self.lookback,
                            ..Default::default()
                        },
                    )
                    .over([col("symbol")])
                    .alias("value_at_risk"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("value_at_risk")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_at_risk_lookback() {
        let factor = ValueAtRisk::with_lookback(126);
        assert_eq!(factor.lookback(), 126);
    }

    #[test]
    fn test_value_at_risk_metadata() {
        let factor = ValueAtRisk::new();
        assert_eq!(factor.name(), "value_at_risk");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

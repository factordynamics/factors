//! Parkinson volatility factor - range-based volatility estimator.
//!
//! Parkinson volatility is a more efficient volatility estimator than standard
//! historical volatility because it uses intraday high and low prices. It assumes
//! no drift and is more efficient than close-to-close volatility.
//!
//! Formula: `σ_P = sqrt((1 / (4 * ln(2))) * mean((ln(H/L))²))`
//!
//! Lower Parkinson volatility indicates tighter trading ranges and lower intraday risk.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Parkinson volatility factor.
///
/// Computes range-based volatility using high and low prices. This is a more
/// efficient estimator than standard deviation of returns when high/low data
/// is available.
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `high`: Daily high price
/// - `low`: Daily low price
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `parkinson_volatility`
#[derive(Debug, Clone)]
pub struct ParkinsonVolatility {
    lookback: usize,
}

impl ParkinsonVolatility {
    /// Create a new ParkinsonVolatility factor with default lookback (20 days).
    pub const fn new() -> Self {
        Self { lookback: 20 }
    }

    /// Create a ParkinsonVolatility factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for ParkinsonVolatility {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for ParkinsonVolatility {
    fn name(&self) -> &str {
        "parkinson_volatility"
    }

    fn description(&self) -> &str {
        "Range-based volatility estimator using high/low prices - more efficient than close-to-close volatility"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Volatility
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "high", "low"]
    }

    fn lookback(&self) -> usize {
        self.lookback
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Parkinson constant: 1 / (4 * ln(2))
        const PARKINSON_CONSTANT: f64 = 1.0 / (4.0 * std::f64::consts::LN_2);
        // Annualization factor for daily to annual volatility
        const TRADING_DAYS_PER_YEAR: f64 = 252.0;
        let annualization_factor = TRADING_DAYS_PER_YEAR.sqrt();

        // Filter to dates up to and including the target date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())));

        // Sort by symbol and date
        let sorted = filtered.sort(
            ["symbol", "date"],
            SortMultipleOptions::default().with_order_descending_multi([false, false]),
        );

        // Compute log(high/low) squared
        let with_hl_ratio = sorted.with_column(
            (col("high") / col("low"))
                .log(std::f64::consts::E)
                .pow(2.0)
                .alias("hl_log_squared"),
        );

        // Compute Parkinson volatility: sqrt(PARKINSON_CONSTANT * mean(hl_log_squared))
        let result = with_hl_ratio
            .with_column(
                col("hl_log_squared")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("hl_mean_squared"),
            )
            .with_column(
                (lit(PARKINSON_CONSTANT) * col("hl_mean_squared"))
                    .sqrt()
                    .alias("daily_parkinson"),
            )
            // Annualize the volatility
            .with_column(
                (col("daily_parkinson") * lit(annualization_factor)).alias("parkinson_volatility"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("parkinson_volatility")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parkinson_volatility_lookback() {
        let factor = ParkinsonVolatility::with_lookback(30);
        assert_eq!(factor.lookback(), 30);
    }

    #[test]
    fn test_parkinson_volatility_metadata() {
        let factor = ParkinsonVolatility::new();
        assert_eq!(factor.name(), "parkinson_volatility");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.lookback(), 20);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "high", "low"]
        );
    }
}

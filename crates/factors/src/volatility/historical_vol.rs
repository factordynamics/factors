//! Historical volatility factor - total risk measure.
//!
//! Historical volatility measures the standard deviation of returns over a
//! lookback period, annualized by scaling with `sqrt(252)`.
//!
//! Higher volatility indicates greater price variability and total risk.
//! Unlike beta, this captures both systematic and idiosyncratic risk.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Historical volatility factor.
///
/// Computes the annualized standard deviation of daily returns over a
/// lookback period. Uses 63 trading days (1 quarter) by default.
///
/// Formula: `σ_annual = std(daily_returns) × sqrt(252)`
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `historical_volatility`
#[derive(Debug, Clone)]
pub struct HistoricalVolatility {
    lookback: usize,
}

impl HistoricalVolatility {
    /// Create a new HistoricalVolatility factor with default lookback (63 days).
    pub const fn new() -> Self {
        Self { lookback: 63 }
    }

    /// Create a HistoricalVolatility factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for HistoricalVolatility {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for HistoricalVolatility {
    fn name(&self) -> &str {
        "historical_volatility"
    }

    fn description(&self) -> &str {
        "Annualized standard deviation of daily returns - total risk measure"
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
        // Annualization factor for daily to annual volatility
        const TRADING_DAYS_PER_YEAR: f64 = 252.0;
        let annualization_factor = TRADING_DAYS_PER_YEAR.sqrt();

        // Filter to dates up to and including the target date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())));

        // Compute daily returns using shift
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

        // Compute rolling standard deviation
        let result = with_returns
            .with_column(
                col("return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("daily_vol"),
            )
            .with_column(
                (col("daily_vol") * lit(annualization_factor)).alias("historical_volatility"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("historical_volatility")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_volatility_lookback() {
        let factor = HistoricalVolatility::with_lookback(21);
        assert_eq!(factor.lookback(), 21);
    }

    #[test]
    fn test_historical_volatility_metadata() {
        let factor = HistoricalVolatility::new();
        assert_eq!(factor.name(), "historical_volatility");
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

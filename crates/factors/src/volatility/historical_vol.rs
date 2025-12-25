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
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for the HistoricalVolatility factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistoricalVolatilityConfig {
    /// Number of trading days for the rolling calculation.
    pub lookback: usize,
    /// Minimum number of periods required for a valid calculation.
    pub min_periods: usize,
}

impl Default for HistoricalVolatilityConfig {
    fn default() -> Self {
        Self {
            lookback: 63,
            min_periods: 63,
        }
    }
}

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
    config: HistoricalVolatilityConfig,
}

impl HistoricalVolatility {
    /// Create a new HistoricalVolatility factor with default lookback (63 days).
    pub fn new() -> Self {
        Self {
            config: HistoricalVolatilityConfig::default(),
        }
    }

    /// Create a HistoricalVolatility factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: HistoricalVolatilityConfig {
                lookback,
                min_periods: lookback,
            },
        }
    }
}

impl Default for HistoricalVolatility {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigurableFactor for HistoricalVolatility {
    type Config = HistoricalVolatilityConfig;

    fn with_config(config: Self::Config) -> Self {
        Self { config }
    }

    fn config(&self) -> &Self::Config {
        &self.config
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
        self.config.lookback
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
                        window_size: self.config.lookback,
                        min_periods: self.config.min_periods,
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

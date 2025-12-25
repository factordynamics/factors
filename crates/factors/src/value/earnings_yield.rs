//! Earnings yield value factor.
//!
//! Measures the ratio of net income to market capitalization.
//! Higher values indicate potentially undervalued securities.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for earnings yield factor.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct EarningsYieldConfig;

impl Default for EarningsYieldConfig {
    fn default() -> Self {
        Self
    }
}

/// Earnings yield value factor.
///
/// Computes the ratio of net income (trailing twelve months) to market capitalization.
/// This is the inverse of the price-to-earnings ratio and captures the "value premium".
/// Higher earnings yield indicates cheaper securities relative to earnings power.
///
/// # Formula
///
/// ```text
/// EarningsYield = Net Income / Market Cap
/// ```
///
/// # Data Requirements
///
/// - `net_income`: Net income (quarterly)
/// - `market_cap`: Market capitalization (daily)
/// - `symbol`: Security identifier
/// - `date`: Date of observation
///
/// # Example
///
/// ```rust,ignore
/// use fd_factors::{Factor, value::EarningsYield};
/// use chrono::NaiveDate;
/// use polars::prelude::*;
///
/// let factor = EarningsYield::default();
/// let data = df![
///     "symbol" => ["AAPL", "MSFT"],
///     "date" => ["2024-03-31", "2024-03-31"],
///     "net_income" => [100_000_000_000.0, 85_000_000_000.0],
///     "market_cap" => [2_500_000_000_000.0, 2_800_000_000_000.0],
/// ]?.lazy();
///
/// let result = factor.compute(&data, NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EarningsYield {
    config: EarningsYieldConfig,
}

impl Default for EarningsYield {
    fn default() -> Self {
        Self {
            config: EarningsYieldConfig,
        }
    }
}

impl Factor for EarningsYield {
    fn name(&self) -> &str {
        "earnings_yield"
    }

    fn description(&self) -> &str {
        "Net income divided by market capitalization - inverse of P/E ratio"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Value
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "net_income", "market_cap"]
    }

    fn lookback(&self) -> usize {
        1
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Quarterly
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let result = data
            .clone()
            .filter(col("date").eq(lit(date.to_string())))
            .select([
                col("symbol"),
                col("date"),
                (col("net_income") / col("market_cap")).alias(self.name()),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for EarningsYield {
    type Config = EarningsYieldConfig;

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
    fn test_earnings_yield_basic() {
        let data = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "net_income" => [100_000_000_000.0, 85_000_000_000.0, 70_000_000_000.0],
            "market_cap" => [2_500_000_000_000.0, 2_500_000_000_000.0, 2_000_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = EarningsYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 3);

        let values = result.column("earnings_yield").unwrap().f64().unwrap();
        assert!((values.get(0).unwrap() - 0.04).abs() < 1e-6);
        assert!((values.get(1).unwrap() - 0.034).abs() < 1e-6);
        assert!((values.get(2).unwrap() - 0.035).abs() < 1e-6);
    }

    #[test]
    fn test_earnings_yield_metadata() {
        let factor = EarningsYield::default();
        assert_eq!(factor.name(), "earnings_yield");
        assert_eq!(factor.category(), FactorCategory::Value);
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
    }

    #[test]
    fn test_earnings_yield_negative_income() {
        let data = df![
            "symbol" => ["AAPL", "TSLA"],
            "date" => ["2024-03-31", "2024-03-31"],
            "net_income" => [100_000_000_000.0, -5_000_000_000.0],
            "market_cap" => [2_500_000_000_000.0, 500_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = EarningsYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        let values = result.column("earnings_yield").unwrap().f64().unwrap();
        assert!(values.get(0).unwrap() > 0.0);
        assert!(values.get(1).unwrap() < 0.0); // Negative earnings yield
    }

    #[test]
    fn test_earnings_yield_date_filtering() {
        let data = df![
            "symbol" => ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date" => ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
            "net_income" => [100_000_000_000.0, 105_000_000_000.0, 85_000_000_000.0, 88_000_000_000.0],
            "market_cap" => [2_500_000_000_000.0, 2_600_000_000_000.0, 2_500_000_000_000.0, 2_550_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = EarningsYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        // Should only get March 31 data
        assert_eq!(result.height(), 2);
        let symbols = result.column("symbol").unwrap().str().unwrap();
        assert_eq!(symbols.get(0).unwrap(), "AAPL");
        assert_eq!(symbols.get(1).unwrap(), "MSFT");
    }
}

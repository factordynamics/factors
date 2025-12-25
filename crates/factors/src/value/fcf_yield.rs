//! Free cash flow yield value factor.
//!
//! Measures the ratio of free cash flow to market capitalization.
//! Higher values indicate potentially undervalued securities.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for free cash flow yield factor.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct FcfYieldConfig;

impl Default for FcfYieldConfig {
    fn default() -> Self {
        Self
    }
}

/// Free cash flow yield value factor.
///
/// Computes the ratio of free cash flow to market capitalization.
/// FCF is often considered a better measure of value than earnings because
/// it represents actual cash generated that can be distributed to shareholders.
///
/// # Formula
///
/// ```text
/// FcfYield = Free Cash Flow / Market Cap
/// ```
///
/// where Free Cash Flow is typically defined as:
/// ```text
/// FCF = Operating Cash Flow - Capital Expenditures
/// ```
///
/// # Data Requirements
///
/// - `free_cash_flow`: Free cash flow (quarterly)
/// - `market_cap`: Market capitalization (daily)
/// - `symbol`: Security identifier
/// - `date`: Date of observation
///
/// # Example
///
/// ```rust,ignore
/// use fd_factors::{Factor, value::FcfYield};
/// use chrono::NaiveDate;
/// use polars::prelude::*;
///
/// let factor = FcfYield::default();
/// let data = df![
///     "symbol" => ["AAPL", "MSFT"],
///     "date" => ["2024-03-31", "2024-03-31"],
///     "free_cash_flow" => [95_000_000_000.0, 70_000_000_000.0],
///     "market_cap" => [2_500_000_000_000.0, 2_800_000_000_000.0],
/// ]?.lazy();
///
/// let result = factor.compute(&data, NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FcfYield {
    config: FcfYieldConfig,
}

impl Default for FcfYield {
    fn default() -> Self {
        Self {
            config: FcfYieldConfig,
        }
    }
}

impl Factor for FcfYield {
    fn name(&self) -> &str {
        "fcf_yield"
    }

    fn description(&self) -> &str {
        "Free cash flow divided by market capitalization - cash generation relative to valuation"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Value
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "free_cash_flow", "market_cap"]
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
                (col("free_cash_flow") / col("market_cap")).alias(self.name()),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for FcfYield {
    type Config = FcfYieldConfig;

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
    fn test_fcf_yield_basic() {
        let data = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "free_cash_flow" => [95_000_000_000.0, 70_000_000_000.0, 60_000_000_000.0],
            "market_cap" => [2_500_000_000_000.0, 2_500_000_000_000.0, 2_000_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = FcfYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 3);

        let values = result.column("fcf_yield").unwrap().f64().unwrap();
        assert!((values.get(0).unwrap() - 0.038).abs() < 1e-6);
        assert!((values.get(1).unwrap() - 0.028).abs() < 1e-6);
        assert!((values.get(2).unwrap() - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_fcf_yield_metadata() {
        let factor = FcfYield::default();
        assert_eq!(factor.name(), "fcf_yield");
        assert_eq!(factor.category(), FactorCategory::Value);
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
    }

    #[test]
    fn test_fcf_yield_negative_fcf() {
        let data = df![
            "symbol" => ["AAPL", "UNPROFITABLE"],
            "date" => ["2024-03-31", "2024-03-31"],
            "free_cash_flow" => [95_000_000_000.0, -10_000_000_000.0],
            "market_cap" => [2_500_000_000_000.0, 500_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = FcfYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        let values = result.column("fcf_yield").unwrap().f64().unwrap();
        assert!(values.get(0).unwrap() > 0.0);
        assert!(values.get(1).unwrap() < 0.0); // Negative FCF yield
    }

    #[test]
    fn test_fcf_yield_date_filtering() {
        let data = df![
            "symbol" => ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date" => ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
            "free_cash_flow" => [95_000_000_000.0, 98_000_000_000.0, 70_000_000_000.0, 73_000_000_000.0],
            "market_cap" => [2_500_000_000_000.0, 2_600_000_000_000.0, 2_500_000_000_000.0, 2_550_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = FcfYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        // Should only get March 31 data
        assert_eq!(result.height(), 2);
        let symbols = result.column("symbol").unwrap().str().unwrap();
        assert_eq!(symbols.get(0).unwrap(), "AAPL");
        assert_eq!(symbols.get(1).unwrap(), "MSFT");
    }

    #[test]
    fn test_fcf_yield_high_growth_company() {
        // High-growth companies often have lower FCF yield
        let data = df![
            "symbol" => ["MATURE", "GROWTH"],
            "date" => ["2024-03-31", "2024-03-31"],
            "free_cash_flow" => [50_000_000_000.0, 5_000_000_000.0],
            "market_cap" => [500_000_000_000.0, 500_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = FcfYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        let values = result.column("fcf_yield").unwrap().f64().unwrap();
        let mature_yield = values.get(0).unwrap();
        let growth_yield = values.get(1).unwrap();

        // Mature company should have higher FCF yield
        assert!(mature_yield > growth_yield);
        assert!((mature_yield - 0.1).abs() < 1e-6);
        assert!((growth_yield - 0.01).abs() < 1e-6);
    }
}

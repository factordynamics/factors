//! Dividend yield value factor.
//!
//! Measures the ratio of dividends per share to stock price.
//! Higher values indicate potentially undervalued securities with strong dividend payouts.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for dividend yield factor.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct DividendYieldConfig;

impl Default for DividendYieldConfig {
    fn default() -> Self {
        Self
    }
}

/// Dividend yield value factor.
///
/// Computes the ratio of dividends per share to stock price.
/// This factor captures the income component of returns and identifies securities
/// that provide attractive dividend yields relative to their price.
///
/// # Formula
///
/// ```text
/// DividendYield = Dividends per Share / Price
/// ```
///
/// # Data Requirements
///
/// - `dividends_per_share`: Dividends paid per share (quarterly)
/// - `close`: Stock price (daily)
/// - `symbol`: Security identifier
/// - `date`: Date of observation
///
/// # Example
///
/// ```rust,ignore
/// use fd_factors::{Factor, value::DividendYield};
/// use chrono::NaiveDate;
/// use polars::prelude::*;
///
/// let factor = DividendYield::default();
/// let data = df![
///     "symbol" => ["AAPL", "MSFT"],
///     "date" => ["2024-03-31", "2024-03-31"],
///     "dividends_per_share" => [0.25, 0.75],
///     "close" => [175.0, 420.0],
/// ]?.lazy();
///
/// let result = factor.compute(&data, NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DividendYield {
    config: DividendYieldConfig,
}

impl Default for DividendYield {
    fn default() -> Self {
        Self {
            config: DividendYieldConfig,
        }
    }
}

impl Factor for DividendYield {
    fn name(&self) -> &str {
        "dividend_yield"
    }

    fn description(&self) -> &str {
        "Dividends per share divided by price - measures income return potential"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Value
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "dividends_per_share", "close"]
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
                (col("dividends_per_share") / col("close")).alias(self.name()),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for DividendYield {
    type Config = DividendYieldConfig;

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
    fn test_dividend_yield_basic() {
        let data = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "dividends_per_share" => [0.25, 0.75, 0.0],
            "close" => [175.0, 420.0, 140.0],
        ]
        .unwrap()
        .lazy();

        let factor = DividendYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 3);

        let values = result.column("dividend_yield").unwrap().f64().unwrap();
        assert!((values.get(0).unwrap() - 0.00142857).abs() < 1e-6); // 0.25 / 175
        assert!((values.get(1).unwrap() - 0.00178571).abs() < 1e-6); // 0.75 / 420
        assert!((values.get(2).unwrap() - 0.0).abs() < 1e-6); // 0.0 / 140
    }

    #[test]
    fn test_dividend_yield_metadata() {
        let factor = DividendYield::default();
        assert_eq!(factor.name(), "dividend_yield");
        assert_eq!(factor.category(), FactorCategory::Value);
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
    }

    #[test]
    fn test_dividend_yield_date_filtering() {
        let data = df![
            "symbol" => ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date" => ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
            "dividends_per_share" => [0.25, 0.30, 0.75, 0.80],
            "close" => [175.0, 180.0, 420.0, 430.0],
        ]
        .unwrap()
        .lazy();

        let factor = DividendYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        // Should only get March 31 data
        assert_eq!(result.height(), 2);
        let symbols = result.column("symbol").unwrap().str().unwrap();
        assert_eq!(symbols.get(0).unwrap(), "AAPL");
        assert_eq!(symbols.get(1).unwrap(), "MSFT");
    }
}

//! Enterprise yield value factor.
//!
//! Measures the ratio of EBIT to enterprise value.
//! Higher values indicate potentially undervalued securities based on operating income.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for enterprise yield factor.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct EnterpriseYieldConfig;

impl Default for EnterpriseYieldConfig {
    fn default() -> Self {
        Self
    }
}

/// Enterprise yield value factor.
///
/// Computes the ratio of EBIT (Earnings Before Interest and Taxes) to enterprise value.
/// This factor provides a capital structure-neutral measure of profitability,
/// similar to the earnings yield but applied to enterprise value instead of equity value.
///
/// # Formula
///
/// ```text
/// EnterpriseYield = EBIT / Enterprise Value
/// ```
///
/// # Data Requirements
///
/// - `ebit`: Earnings before interest and taxes (quarterly)
/// - `enterprise_value`: Market cap + debt - cash (daily)
/// - `symbol`: Security identifier
/// - `date`: Date of observation
///
/// # Example
///
/// ```rust,ignore
/// use fd_factors::{Factor, value::EnterpriseYield};
/// use chrono::NaiveDate;
/// use polars::prelude::*;
///
/// let factor = EnterpriseYield::default();
/// let data = df![
///     "symbol" => ["AAPL", "MSFT"],
///     "date" => ["2024-03-31", "2024-03-31"],
///     "ebit" => [30_000_000_000.0, 25_000_000_000.0],
///     "enterprise_value" => [2_450_000_000_000.0, 2_750_000_000_000.0],
/// ]?.lazy();
///
/// let result = factor.compute(&data, NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EnterpriseYield {
    config: EnterpriseYieldConfig,
}

impl Default for EnterpriseYield {
    fn default() -> Self {
        Self {
            config: EnterpriseYieldConfig,
        }
    }
}

impl Factor for EnterpriseYield {
    fn name(&self) -> &str {
        "enterprise_yield"
    }

    fn description(&self) -> &str {
        "EBIT divided by enterprise value - measures operating income relative to firm value"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Value
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "ebit", "enterprise_value"]
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
                (col("ebit") / col("enterprise_value")).alias(self.name()),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for EnterpriseYield {
    type Config = EnterpriseYieldConfig;

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
    fn test_enterprise_yield_basic() {
        let data = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "ebit" => [30_000_000_000.0, 25_000_000_000.0, 22_000_000_000.0],
            "enterprise_value" => [2_450_000_000_000.0, 2_750_000_000_000.0, 1_950_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = EnterpriseYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 3);

        let values = result.column("enterprise_yield").unwrap().f64().unwrap();
        assert!((values.get(0).unwrap() - 0.012244898).abs() < 1e-6); // 30B / 2450B
        assert!((values.get(1).unwrap() - 0.009090909).abs() < 1e-6); // 25B / 2750B
        assert!((values.get(2).unwrap() - 0.011282051).abs() < 1e-6); // 22B / 1950B
    }

    #[test]
    fn test_enterprise_yield_metadata() {
        let factor = EnterpriseYield::default();
        assert_eq!(factor.name(), "enterprise_yield");
        assert_eq!(factor.category(), FactorCategory::Value);
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
    }

    #[test]
    fn test_enterprise_yield_date_filtering() {
        let data = df![
            "symbol" => ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date" => ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
            "ebit" => [30_000_000_000.0, 31_000_000_000.0, 25_000_000_000.0, 26_000_000_000.0],
            "enterprise_value" => [2_450_000_000_000.0, 2_500_000_000_000.0, 2_750_000_000_000.0, 2_800_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = EnterpriseYield::default();
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        // Should only get March 31 data
        assert_eq!(result.height(), 2);
        let symbols = result.column("symbol").unwrap().str().unwrap();
        assert_eq!(symbols.get(0).unwrap(), "AAPL");
        assert_eq!(symbols.get(1).unwrap(), "MSFT");
    }
}

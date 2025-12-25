//! Enterprise value factor.
//!
//! Computes enterprise value, which represents a company's total value including debt.
//! Enterprise value is a more comprehensive measure of company value than market cap alone.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for enterprise value factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnterpriseValueConfig {
    // Empty for now - size factors typically don't need configuration
    // This struct exists for consistency and future extensibility
}

/// Enterprise value factor.
///
/// This factor measures company size using enterprise value (EV).
/// EV represents the theoretical takeover price of a company, accounting for
/// both equity value and net debt.
///
/// # Formula
///
/// `enterprise_value = market_cap + total_debt - cash`
///
/// # Properties
///
/// - **Category**: Size
/// - **Frequency**: Quarterly (requires fundamental data)
/// - **Lookback**: 1 quarter (current values only)
/// - **Required columns**: `["symbol", "date", "market_cap", "total_debt", "cash"]`
///
/// # Example
///
/// ```ignore
/// use fd_factors::{Factor, size::EnterpriseValue};
/// use polars::prelude::*;
/// use chrono::NaiveDate;
///
/// let factor = EnterpriseValue::default();
/// let data = df![
///     "symbol" => ["AAPL", "MSFT"],
///     "date" => ["2024-01-01", "2024-01-01"],
///     "market_cap" => [2_400_000_000_000.0, 2_250_000_000_000.0],
///     "total_debt" => [100_000_000_000.0, 50_000_000_000.0],
///     "cash" => [50_000_000_000.0, 100_000_000_000.0],
/// ]?.lazy();
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let result = factor.compute(&data, date)?;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EnterpriseValue {
    config: EnterpriseValueConfig,
}

impl Factor for EnterpriseValue {
    fn name(&self) -> &str {
        "enterprise_value"
    }

    fn description(&self) -> &str {
        "Enterprise value (market cap + total debt - cash)"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Size
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "market_cap", "total_debt", "cash"]
    }

    fn lookback(&self) -> usize {
        1
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Quarterly
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let date_str = date.format("%Y-%m-%d").to_string();

        let result = data
            .clone()
            .filter(col("date").eq(lit(date_str)))
            .select([
                col("symbol"),
                col("date"),
                (col("market_cap") + col("total_debt") - col("cash")).alias("enterprise_value"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for EnterpriseValue {
    type Config = EnterpriseValueConfig;

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
    use approx::assert_relative_eq;

    #[test]
    fn test_enterprise_value_computation() {
        // Create test data
        let df = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-01-01", "2024-01-01", "2024-01-01"],
            "market_cap" => [2_400_000_000_000.0, 2_250_000_000_000.0, 1_000_000_000_000.0],
            "total_debt" => [100_000_000_000.0, 50_000_000_000.0, 20_000_000_000.0],
            "cash" => [50_000_000_000.0, 100_000_000_000.0, 80_000_000_000.0],
        ]
        .unwrap();

        let factor = EnterpriseValue::default();
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let result = factor.compute_raw(&df.lazy(), date).unwrap();

        // Extract results
        let symbols = result
            .column("symbol")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>();
        let evs = result
            .column("enterprise_value")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>();

        // Verify calculations
        // AAPL: 2.4T + 100B - 50B = 2.45T
        // MSFT: 2.25T + 50B - 100B = 2.2T
        // GOOGL: 1T + 20B - 80B = 940B

        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0], "AAPL");
        assert_eq!(symbols[1], "MSFT");
        assert_eq!(symbols[2], "GOOGL");

        assert_relative_eq!(
            evs[0],
            2_400_000_000_000.0 + 100_000_000_000.0 - 50_000_000_000.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            evs[1],
            2_250_000_000_000.0 + 50_000_000_000.0 - 100_000_000_000.0,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            evs[2],
            1_000_000_000_000.0 + 20_000_000_000.0 - 80_000_000_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_enterprise_value_filters_by_date() {
        // Create test data with multiple dates
        let df = df![
            "symbol" => ["AAPL", "AAPL", "MSFT"],
            "date" => ["2024-01-01", "2024-04-01", "2024-01-01"],
            "market_cap" => [2_400_000_000_000.0, 2_500_000_000_000.0, 2_250_000_000_000.0],
            "total_debt" => [100_000_000_000.0, 110_000_000_000.0, 50_000_000_000.0],
            "cash" => [50_000_000_000.0, 60_000_000_000.0, 100_000_000_000.0],
        ]
        .unwrap();

        let factor = EnterpriseValue::default();
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let result = factor.compute_raw(&df.lazy(), date).unwrap();

        // Should only return data for 2024-01-01
        assert_eq!(result.height(), 2);

        let symbols = result
            .column("symbol")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>();

        assert_eq!(symbols[0], "AAPL");
        assert_eq!(symbols[1], "MSFT");
    }

    #[test]
    fn test_enterprise_value_trait_properties() {
        let factor = EnterpriseValue::default();

        assert_eq!(factor.name(), "enterprise_value");
        assert_eq!(
            factor.description(),
            "Enterprise value (market cap + total debt - cash)"
        );
        assert_eq!(factor.category(), FactorCategory::Size);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "market_cap", "total_debt", "cash"]
        );
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
    }

    #[test]
    fn test_enterprise_value_output_columns() {
        let df = df![
            "symbol" => ["AAPL"],
            "date" => ["2024-01-01"],
            "market_cap" => [2_400_000_000_000.0],
            "total_debt" => [100_000_000_000.0],
            "cash" => [50_000_000_000.0],
        ]
        .unwrap();

        let factor = EnterpriseValue::default();
        let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let result = factor.compute_raw(&df.lazy(), date).unwrap();

        // Verify output schema
        let schema = result.schema();
        assert!(schema.contains("symbol"));
        assert!(schema.contains("date"));
        assert!(schema.contains("enterprise_value"));
        assert_eq!(schema.len(), 3);
    }
}

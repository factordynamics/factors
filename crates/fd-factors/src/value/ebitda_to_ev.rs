//! EBITDA-to-EV value factor.
//!
//! Measures the ratio of EBITDA to enterprise value.
//! Higher values indicate potentially undervalued securities based on operating earnings.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// EBITDA-to-EV value factor.
///
/// Computes the ratio of EBITDA (Earnings Before Interest, Taxes, Depreciation,
/// and Amortization) to enterprise value. This factor provides a capital structure-neutral
/// measure of operating profitability relative to firm value.
///
/// # Formula
///
/// ```text
/// EbitdaToEv = EBITDA / Enterprise Value
/// ```
///
/// # Data Requirements
///
/// - `ebitda`: Earnings before interest, taxes, depreciation, and amortization (quarterly)
/// - `enterprise_value`: Market cap + debt - cash (daily)
/// - `symbol`: Security identifier
/// - `date`: Date of observation
///
/// # Example
///
/// ```rust,ignore
/// use fd_factors::{Factor, value::EbitdaToEv};
/// use chrono::NaiveDate;
/// use polars::prelude::*;
///
/// let factor = EbitdaToEv::default();
/// let data = df![
///     "symbol" => ["AAPL", "MSFT"],
///     "date" => ["2024-03-31", "2024-03-31"],
///     "ebitda" => [35_000_000_000.0, 30_000_000_000.0],
///     "enterprise_value" => [2_450_000_000_000.0, 2_750_000_000_000.0],
/// ]?.lazy();
///
/// let result = factor.compute(&data, NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())?;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct EbitdaToEv;

impl Factor for EbitdaToEv {
    fn name(&self) -> &str {
        "ebitda_to_ev"
    }

    fn description(&self) -> &str {
        "EBITDA divided by enterprise value - measures operating profitability relative to firm value"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Value
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "ebitda", "enterprise_value"]
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
                (col("ebitda") / col("enterprise_value")).alias(self.name()),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ebitda_to_ev_basic() {
        let data = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "ebitda" => [35_000_000_000.0, 30_000_000_000.0, 28_000_000_000.0],
            "enterprise_value" => [2_450_000_000_000.0, 2_750_000_000_000.0, 1_950_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = EbitdaToEv;
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 3);

        let values = result.column("ebitda_to_ev").unwrap().f64().unwrap();
        assert!((values.get(0).unwrap() - 0.014285714).abs() < 1e-6); // 35B / 2450B
        assert!((values.get(1).unwrap() - 0.010909091).abs() < 1e-6); // 30B / 2750B
        assert!((values.get(2).unwrap() - 0.014358974).abs() < 1e-6); // 28B / 1950B
    }

    #[test]
    fn test_ebitda_to_ev_metadata() {
        let factor = EbitdaToEv;
        assert_eq!(factor.name(), "ebitda_to_ev");
        assert_eq!(factor.category(), FactorCategory::Value);
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
    }

    #[test]
    fn test_ebitda_to_ev_date_filtering() {
        let data = df![
            "symbol" => ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date" => ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
            "ebitda" => [35_000_000_000.0, 36_000_000_000.0, 30_000_000_000.0, 31_000_000_000.0],
            "enterprise_value" => [2_450_000_000_000.0, 2_500_000_000_000.0, 2_750_000_000_000.0, 2_800_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = EbitdaToEv;
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        // Should only get March 31 data
        assert_eq!(result.height(), 2);
        let symbols = result.column("symbol").unwrap().str().unwrap();
        assert_eq!(symbols.get(0).unwrap(), "AAPL");
        assert_eq!(symbols.get(1).unwrap(), "MSFT");
    }
}

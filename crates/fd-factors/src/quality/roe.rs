//! Return on Equity (ROE) factor.
//!
//! ROE measures a company's profitability relative to shareholder equity.
//! Higher ROE suggests efficient use of equity capital to generate profits.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Return on Equity factor.
///
/// ROE is calculated as:
/// ```text
/// ROE = Net Income / Shareholders' Equity
/// ```
///
/// This factor is commonly used in quality-based strategies and indicates
/// how well a company generates profits from shareholder investments.
#[derive(Debug, Clone, Default)]
pub struct Roe;

impl Factor for Roe {
    fn name(&self) -> &str {
        "roe"
    }

    fn description(&self) -> &str {
        "Return on Equity - net income divided by shareholders' equity"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "net_income", "shareholders_equity"]
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
                (col("net_income") / col("shareholders_equity")).alias("roe"),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roe_metadata() {
        let factor = Roe;
        assert_eq!(factor.name(), "roe");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_roe_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "net_income" => [25000.0, 15000.0, 20000.0],
            "shareholders_equity" => [100000.0, 150000.0, 200000.0]
        ]
        .unwrap();

        let factor = Roe;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let roe_values = result.column("roe").unwrap().f64().unwrap();
        assert!((roe_values.get(0).unwrap() - 0.25).abs() < 1e-6); // 25000/100000
        assert!((roe_values.get(1).unwrap() - 0.10).abs() < 1e-6); // 15000/150000
        assert!((roe_values.get(2).unwrap() - 0.10).abs() < 1e-6); // 20000/200000
    }
}

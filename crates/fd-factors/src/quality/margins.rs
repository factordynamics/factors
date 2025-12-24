//! Profit Margin factor.
//!
//! Profit margin measures the percentage of revenue that becomes profit.
//! Higher margins indicate better cost control and pricing power.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Profit Margin factor.
///
/// Profit margin is calculated as:
/// ```text
/// Profit Margin = Net Income / Revenue
/// ```
///
/// This factor is used in quality-based strategies to identify companies
/// with strong operational efficiency and pricing power.
#[derive(Debug, Clone, Default)]
pub struct ProfitMargin;

impl Factor for ProfitMargin {
    fn name(&self) -> &str {
        "profit_margin"
    }

    fn description(&self) -> &str {
        "Profit Margin - net income divided by revenue"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "net_income", "revenue"]
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
                (col("net_income") / col("revenue")).alias("profit_margin"),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profit_margin_metadata() {
        let factor = ProfitMargin;
        assert_eq!(factor.name(), "profit_margin");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_profit_margin_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "net_income" => [25000.0, 15000.0, 20000.0],
            "revenue" => [100000.0, 75000.0, 80000.0]
        ]
        .unwrap();

        let factor = ProfitMargin;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let margin_values = result.column("profit_margin").unwrap().f64().unwrap();
        assert!((margin_values.get(0).unwrap() - 0.25).abs() < 1e-6); // 25000/100000
        assert!((margin_values.get(1).unwrap() - 0.20).abs() < 1e-6); // 15000/75000
        assert!((margin_values.get(2).unwrap() - 0.25).abs() < 1e-6); // 20000/80000
    }
}

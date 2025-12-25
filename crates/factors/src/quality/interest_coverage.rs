//! Interest Coverage factor.
//!
//! Interest Coverage measures a company's ability to pay interest on its debt.
//! Higher coverage ratios indicate better financial health and lower default risk.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Interest Coverage factor.
///
/// Interest Coverage is calculated as:
/// ```text
/// Interest Coverage = EBIT / Interest Expense
/// ```
///
/// This factor measures how many times a company can cover its interest payments
/// with its operating earnings. Higher ratios indicate stronger financial health.
#[derive(Debug, Clone, Default)]
pub struct InterestCoverage;

impl Factor for InterestCoverage {
    fn name(&self) -> &str {
        "interest_coverage"
    }

    fn description(&self) -> &str {
        "Interest Coverage - EBIT divided by interest expense"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "ebit", "interest_expense"]
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
                (col("ebit") / col("interest_expense")).alias("interest_coverage"),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interest_coverage_metadata() {
        let factor = InterestCoverage;
        assert_eq!(factor.name(), "interest_coverage");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_interest_coverage_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "ebit" => [35000.0, 25000.0, 30000.0],
            "interest_expense" => [1000.0, 2500.0, 1500.0]
        ]
        .unwrap();

        let factor = InterestCoverage;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let coverage_values = result.column("interest_coverage").unwrap().f64().unwrap();
        // AAPL: 35000 / 1000 = 35.0
        assert!((coverage_values.get(0).unwrap() - 35.0).abs() < 1e-6);
        // GOOGL: 25000 / 2500 = 10.0
        assert!((coverage_values.get(1).unwrap() - 10.0).abs() < 1e-6);
        // MSFT: 30000 / 1500 = 20.0
        assert!((coverage_values.get(2).unwrap() - 20.0).abs() < 1e-6);
    }
}

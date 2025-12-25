//! Quick Ratio factor.
//!
//! Quick Ratio measures a company's ability to pay short-term obligations with
//! its most liquid assets. Also known as the Acid-Test Ratio.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Quick Ratio factor.
///
/// Quick Ratio is calculated as:
/// ```text
/// Quick Ratio = (Current Assets - Inventory) / Current Liabilities
/// ```
///
/// This factor is a more conservative measure of liquidity than the current ratio,
/// as it excludes inventory which may not be easily converted to cash.
#[derive(Debug, Clone, Default)]
pub struct QuickRatio;

impl Factor for QuickRatio {
    fn name(&self) -> &str {
        "quick_ratio"
    }

    fn description(&self) -> &str {
        "Quick Ratio - (current assets minus inventory) divided by current liabilities"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &[
            "symbol",
            "date",
            "current_assets",
            "inventory",
            "current_liabilities",
        ]
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
                ((col("current_assets") - col("inventory")) / col("current_liabilities"))
                    .alias("quick_ratio"),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_ratio_metadata() {
        let factor = QuickRatio;
        assert_eq!(factor.name(), "quick_ratio");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_quick_ratio_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "current_assets" => [150000.0, 120000.0, 180000.0],
            "inventory" => [30000.0, 20000.0, 30000.0],
            "current_liabilities" => [100000.0, 80000.0, 150000.0]
        ]
        .unwrap();

        let factor = QuickRatio;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let ratio_values = result.column("quick_ratio").unwrap().f64().unwrap();
        // AAPL: (150000 - 30000) / 100000 = 120000 / 100000 = 1.2
        assert!((ratio_values.get(0).unwrap() - 1.2).abs() < 1e-6);
        // GOOGL: (120000 - 20000) / 80000 = 100000 / 80000 = 1.25
        assert!((ratio_values.get(1).unwrap() - 1.25).abs() < 1e-6);
        // MSFT: (180000 - 30000) / 150000 = 150000 / 150000 = 1.0
        assert!((ratio_values.get(2).unwrap() - 1.0).abs() < 1e-6);
    }
}

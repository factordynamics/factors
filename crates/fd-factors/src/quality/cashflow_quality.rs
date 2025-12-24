//! Cash Flow Quality factor.
//!
//! Measures the quality of earnings by comparing operating cash flow to net income.
//! Higher values indicate earnings are more backed by actual cash generation.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Cash Flow Quality factor.
///
/// Calculated as:
/// ```text
/// CFQ = Operating Cash Flow / Net Income
/// ```
///
/// This ratio indicates earnings quality:
/// - Ratio > 1: Cash generation exceeds reported earnings (high quality)
/// - Ratio ≈ 1: Cash generation matches earnings (normal)
/// - Ratio < 1: Earnings exceed cash generation (potential quality issues)
/// - Ratio < 0: Negative earnings or cash flow (distress signal)
///
/// Higher values are generally preferred as they indicate that reported profits
/// are backed by actual cash flows rather than accounting accruals.
#[derive(Debug, Clone, Default)]
pub struct CashflowQuality;

impl Factor for CashflowQuality {
    fn name(&self) -> &str {
        "cashflow_quality"
    }

    fn description(&self) -> &str {
        "Cash Flow Quality - operating cash flow divided by net income"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "operating_cash_flow", "net_income"]
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
                (col("operating_cash_flow") / col("net_income")).alias("cashflow_quality"),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cashflow_quality_metadata() {
        let factor = CashflowQuality;
        assert_eq!(factor.name(), "cashflow_quality");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_cashflow_quality_high_quality() {
        // Cash flow exceeds net income (high quality)
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "operating_cash_flow" => [30000.0, 25000.0, 35000.0],
            "net_income" => [25000.0, 20000.0, 28000.0]
        ]
        .unwrap();

        let factor = CashflowQuality;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let quality = result.column("cashflow_quality").unwrap().f64().unwrap();

        // All should be > 1 (high quality)
        assert!((quality.get(0).unwrap() - 1.2).abs() < 1e-6); // 30000/25000 = 1.2
        assert!((quality.get(1).unwrap() - 1.25).abs() < 1e-6); // 25000/20000 = 1.25
        assert!((quality.get(2).unwrap() - 1.25).abs() < 1e-6); // 35000/28000 = 1.25
    }

    #[test]
    fn test_cashflow_quality_low_quality() {
        // Net income exceeds cash flow (potential quality issues)
        let df = df![
            "symbol" => ["WEAK"],
            "date" => ["2024-03-31"],
            "operating_cash_flow" => [15000.0],
            "net_income" => [25000.0]
        ]
        .unwrap();

        let factor = CashflowQuality;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let quality = result.column("cashflow_quality").unwrap().f64().unwrap();

        // Should be < 1 (lower quality)
        assert!((quality.get(0).unwrap() - 0.6).abs() < 1e-6); // 15000/25000 = 0.6
        assert!(quality.get(0).unwrap() < 1.0);
    }

    #[test]
    fn test_cashflow_quality_negative_income() {
        // Negative net income with positive cash flow
        let df = df![
            "symbol" => ["TURNAROUND"],
            "date" => ["2024-03-31"],
            "operating_cash_flow" => [10000.0],
            "net_income" => [-5000.0]
        ]
        .unwrap();

        let factor = CashflowQuality;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let quality = result.column("cashflow_quality").unwrap().f64().unwrap();

        // Should be negative (10000 / -5000 = -2.0)
        assert!((quality.get(0).unwrap() - (-2.0)).abs() < 1e-6);
        assert!(quality.get(0).unwrap() < 0.0);
    }

    #[test]
    fn test_cashflow_quality_perfect_match() {
        // OCF equals NI (ratio = 1)
        let df = df![
            "symbol" => ["MATCH"],
            "date" => ["2024-03-31"],
            "operating_cash_flow" => [20000.0],
            "net_income" => [20000.0]
        ]
        .unwrap();

        let factor = CashflowQuality;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let quality = result.column("cashflow_quality").unwrap().f64().unwrap();

        // Should be exactly 1.0
        assert!((quality.get(0).unwrap() - 1.0).abs() < 1e-6);
    }
}

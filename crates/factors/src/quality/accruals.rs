//! Accruals Quality factor (Sloan 1996).
//!
//! Accruals Quality measures the difference between reported earnings and cash flows.
//! Companies with high accruals (earnings >> cash flow) tend to underperform, as high
//! accruals suggest potentially lower earnings quality.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Accruals Quality factor.
///
/// Accruals Quality is calculated as:
/// ```text
/// Accruals Quality = Total Accruals / Total Assets
/// where Total Accruals = Net Income - Operating Cash Flow
/// ```
///
/// Lower (more negative) values indicate higher earnings quality, as earnings
/// are backed by actual cash flows. Higher values suggest earnings may be
/// inflated by accounting accruals rather than real cash generation.
///
/// Based on Sloan (1996): "Do Stock Prices Fully Reflect Information in
/// Accruals and Cash Flows about Future Earnings?"
#[derive(Debug, Clone, Default)]
pub struct AccrualsQuality;

impl Factor for AccrualsQuality {
    fn name(&self) -> &str {
        "accruals_quality"
    }

    fn description(&self) -> &str {
        "Accruals quality (Sloan) - earnings vs cash flow divergence"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &[
            "symbol",
            "date",
            "net_income",
            "operating_cash_flow",
            "total_assets",
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
                ((col("net_income") - col("operating_cash_flow")) / col("total_assets"))
                    .alias("accruals_quality"),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accruals_quality_metadata() {
        let factor = AccrualsQuality;
        assert_eq!(factor.name(), "accruals_quality");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_accruals_quality_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "net_income" => [25000.0, 15000.0, 20000.0],
            "operating_cash_flow" => [30000.0, 10000.0, 20000.0],
            "total_assets" => [500000.0, 300000.0, 400000.0]
        ]
        .unwrap();

        let factor = AccrualsQuality;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let accruals_values = result.column("accruals_quality").unwrap().f64().unwrap();

        // AAPL: (25000 - 30000) / 500000 = -5000 / 500000 = -0.01
        // Negative accruals (cash flow > earnings) = high quality
        assert!((accruals_values.get(0).unwrap() - (-0.01)).abs() < 1e-6);

        // GOOGL: (15000 - 10000) / 300000 = 5000 / 300000 = 0.0166...
        // Positive accruals (earnings > cash flow) = lower quality
        assert!((accruals_values.get(1).unwrap() - 0.016666666666666666).abs() < 1e-6);

        // MSFT: (20000 - 20000) / 400000 = 0 / 400000 = 0.0
        // Zero accruals (earnings = cash flow) = perfect alignment
        assert!((accruals_values.get(2).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_accruals_quality_high_accruals() {
        // Test case where earnings far exceed cash flow (low quality)
        let df = df![
            "symbol" => ["XYZ"],
            "date" => ["2024-03-31"],
            "net_income" => [50000.0],
            "operating_cash_flow" => [10000.0],
            "total_assets" => [200000.0]
        ]
        .unwrap();

        let factor = AccrualsQuality;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let accruals_values = result.column("accruals_quality").unwrap().f64().unwrap();

        // (50000 - 10000) / 200000 = 40000 / 200000 = 0.2
        // High positive value indicates low quality (high accruals)
        assert!((accruals_values.get(0).unwrap() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_accruals_quality_negative_accruals() {
        // Test case where cash flow far exceeds earnings (high quality)
        let df = df![
            "symbol" => ["ABC"],
            "date" => ["2024-03-31"],
            "net_income" => [10000.0],
            "operating_cash_flow" => [50000.0],
            "total_assets" => [200000.0]
        ]
        .unwrap();

        let factor = AccrualsQuality;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let accruals_values = result.column("accruals_quality").unwrap().f64().unwrap();

        // (10000 - 50000) / 200000 = -40000 / 200000 = -0.2
        // High negative value indicates high quality (cash-backed earnings)
        assert!((accruals_values.get(0).unwrap() - (-0.2)).abs() < 1e-6);
    }
}

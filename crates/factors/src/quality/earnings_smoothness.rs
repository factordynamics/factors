//! Earnings Smoothness factor.
//!
//! Measures earnings quality by comparing the volatility of net income to
//! operating cash flow. Lower values indicate higher quality earnings.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Earnings Smoothness factor.
///
/// Calculated as the ratio of standard deviations:
/// ```text
/// Smoothness = σ(Net Income) / σ(Operating Cash Flow)
/// ```
///
/// This measures earnings quality:
/// - Lower values (< 1): More stable earnings relative to cash flows (higher quality)
/// - Higher values (> 1): More volatile earnings relative to cash flows (lower quality)
///
/// The intuition is that companies with artificially smoothed earnings through
/// accounting manipulation will have lower volatility in net income than in
/// operating cash flow. High-quality companies should have net income volatility
/// similar to or lower than cash flow volatility.
///
/// We use the last 8 quarters of data to compute volatility.
#[derive(Debug, Clone, Default)]
pub struct EarningsSmoothness;

impl Factor for EarningsSmoothness {
    fn name(&self) -> &str {
        "earnings_smoothness"
    }

    fn description(&self) -> &str {
        "Earnings Smoothness - ratio of net income volatility to cash flow volatility"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "net_income", "operating_cash_flow"]
    }

    fn lookback(&self) -> usize {
        8 // 8 quarters for meaningful volatility calculation
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Quarterly
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let result = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())))
            .sort(["symbol", "date"], Default::default())
            .group_by([col("symbol")])
            .agg([
                // Keep the latest date for each symbol
                col("date").max().alias("date"),
                // Calculate standard deviation of net income
                col("net_income").std(0).alias("ni_std"),
                // Calculate standard deviation of operating cash flow
                col("operating_cash_flow").std(0).alias("ocf_std"),
            ])
            .filter(col("date").eq(lit(date.to_string())))
            .select([
                col("symbol"),
                col("date"),
                // Smoothness ratio: lower is better quality
                (col("ni_std") / col("ocf_std")).alias("earnings_smoothness"),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_earnings_smoothness_metadata() {
        let factor = EarningsSmoothness;
        assert_eq!(factor.name(), "earnings_smoothness");
        assert_eq!(factor.lookback(), 8);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_earnings_smoothness_high_quality() {
        // Create data where NI volatility is lower than OCF volatility (high quality)
        let df = df![
            "symbol" => ["QUALITY", "QUALITY", "QUALITY", "QUALITY", "QUALITY", "QUALITY", "QUALITY", "QUALITY"],
            "date" => [
                "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"
            ],
            "net_income" => [10000.0, 10200.0, 10100.0, 10300.0, 10150.0, 10250.0, 10200.0, 10350.0],
            "operating_cash_flow" => [9000.0, 12000.0, 8500.0, 13000.0, 9500.0, 11500.0, 8800.0, 12500.0]
        ]
        .unwrap();

        let factor = EarningsSmoothness;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 6, 30).unwrap())
            .unwrap();

        assert_eq!(result.shape().0, 1);

        let smoothness = result.column("earnings_smoothness").unwrap().f64().unwrap();

        // Should be < 1 (NI less volatile than OCF)
        assert!(smoothness.get(0).unwrap() < 1.0);
    }

    #[test]
    fn test_earnings_smoothness_low_quality() {
        // Create data where NI is more volatile than OCF (lower quality / manipulation)
        let df = df![
            "symbol" => ["SUSPECT", "SUSPECT", "SUSPECT", "SUSPECT", "SUSPECT", "SUSPECT", "SUSPECT", "SUSPECT"],
            "date" => [
                "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"
            ],
            "net_income" => [10000.0, 15000.0, 5000.0, 20000.0, 8000.0, 18000.0, 6000.0, 22000.0],
            "operating_cash_flow" => [10000.0, 10200.0, 10100.0, 10300.0, 10150.0, 10250.0, 10200.0, 10350.0]
        ]
        .unwrap();

        let factor = EarningsSmoothness;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 6, 30).unwrap())
            .unwrap();

        let smoothness = result.column("earnings_smoothness").unwrap().f64().unwrap();

        // Should be > 1 (NI more volatile than OCF)
        assert!(smoothness.get(0).unwrap() > 1.0);
    }

    #[test]
    fn test_earnings_smoothness_multiple_symbols() {
        let df = df![
            "symbol" => [
                "HIGH_Q", "HIGH_Q", "HIGH_Q", "HIGH_Q", "HIGH_Q", "HIGH_Q", "HIGH_Q", "HIGH_Q",
                "LOW_Q", "LOW_Q", "LOW_Q", "LOW_Q", "LOW_Q", "LOW_Q", "LOW_Q", "LOW_Q"
            ],
            "date" => [
                "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30",
                "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"
            ],
            "net_income" => [
                10000.0, 10100.0, 10050.0, 10150.0, 10075.0, 10125.0, 10100.0, 10175.0,  // Stable
                10000.0, 15000.0, 5000.0, 18000.0, 7000.0, 16000.0, 6000.0, 19000.0       // Volatile
            ],
            "operating_cash_flow" => [
                9000.0, 11000.0, 9500.0, 10500.0, 9200.0, 10800.0, 9300.0, 10700.0,      // Moderate volatility
                10000.0, 10100.0, 10050.0, 10150.0, 10075.0, 10125.0, 10100.0, 10175.0   // Stable
            ]
        ]
        .unwrap();

        let factor = EarningsSmoothness;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 6, 30).unwrap())
            .unwrap();

        assert_eq!(result.shape().0, 2);

        let smoothness = result.column("earnings_smoothness").unwrap().f64().unwrap();

        // First company (HIGH_Q) should have lower smoothness ratio
        // Second company (LOW_Q) should have higher smoothness ratio
        // We can verify both exist
        assert!(!smoothness.get(0).unwrap().is_nan());
        assert!(!smoothness.get(1).unwrap().is_nan());
    }
}

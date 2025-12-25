//! Current Ratio factor.
//!
//! Current Ratio measures a company's ability to pay short-term obligations.
//! Higher ratios indicate better liquidity and financial stability.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Current Ratio factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct CurrentRatioConfig {}

/// Current Ratio factor.
///
/// Current Ratio is calculated as:
/// ```text
/// Current Ratio = Current Assets / Current Liabilities
/// ```
///
/// This factor measures a company's ability to pay short-term obligations with
/// short-term assets. A ratio above 1.0 indicates the company can cover its
/// short-term liabilities.
#[derive(Debug, Clone, Default)]
pub struct CurrentRatio {
    config: CurrentRatioConfig,
}

impl Factor for CurrentRatio {
    fn name(&self) -> &str {
        "current_ratio"
    }

    fn description(&self) -> &str {
        "Current Ratio - current assets divided by current liabilities"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "current_assets", "current_liabilities"]
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
                (col("current_assets") / col("current_liabilities")).alias("current_ratio"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for CurrentRatio {
    type Config = CurrentRatioConfig;

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
    fn test_current_ratio_metadata() {
        let factor = CurrentRatio::default();
        assert_eq!(factor.name(), "current_ratio");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_current_ratio_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "current_assets" => [150000.0, 120000.0, 180000.0],
            "current_liabilities" => [100000.0, 80000.0, 150000.0]
        ]
        .unwrap();

        let factor = CurrentRatio::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let ratio_values = result.column("current_ratio").unwrap().f64().unwrap();
        // AAPL: 150000 / 100000 = 1.5
        assert!((ratio_values.get(0).unwrap() - 1.5).abs() < 1e-6);
        // GOOGL: 120000 / 80000 = 1.5
        assert!((ratio_values.get(1).unwrap() - 1.5).abs() < 1e-6);
        // MSFT: 180000 / 150000 = 1.2
        assert!((ratio_values.get(2).unwrap() - 1.2).abs() < 1e-6);
    }
}

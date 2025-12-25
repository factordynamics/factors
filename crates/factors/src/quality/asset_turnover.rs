//! Asset Turnover factor.
//!
//! Asset Turnover measures how efficiently a company uses its assets to generate revenue.
//! Higher values indicate better asset utilization and operational efficiency.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Asset Turnover factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssetTurnoverConfig {}

/// Asset Turnover factor.
///
/// Asset Turnover is calculated as:
/// ```text
/// Asset Turnover = Revenue / Total Assets
/// ```
///
/// This factor measures how effectively a company uses its assets to generate sales.
/// Higher turnover ratios suggest more efficient use of assets.
#[derive(Debug, Clone, Default)]
pub struct AssetTurnover {
    config: AssetTurnoverConfig,
}

impl Factor for AssetTurnover {
    fn name(&self) -> &str {
        "asset_turnover"
    }

    fn description(&self) -> &str {
        "Asset Turnover - revenue divided by total assets"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "revenue", "total_assets"]
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
                (col("revenue") / col("total_assets")).alias("asset_turnover"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for AssetTurnover {
    type Config = AssetTurnoverConfig;

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
    fn test_asset_turnover_metadata() {
        let factor = AssetTurnover::default();
        assert_eq!(factor.name(), "asset_turnover");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_asset_turnover_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "revenue" => [100000.0, 75000.0, 60000.0],
            "total_assets" => [400000.0, 350000.0, 500000.0]
        ]
        .unwrap();

        let factor = AssetTurnover::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let turnover_values = result.column("asset_turnover").unwrap().f64().unwrap();
        // AAPL: 100000 / 400000 = 0.25
        assert!((turnover_values.get(0).unwrap() - 0.25).abs() < 1e-6);
        // GOOGL: 75000 / 350000 = 0.214285...
        assert!((turnover_values.get(1).unwrap() - 0.214285714).abs() < 1e-6);
        // MSFT: 60000 / 500000 = 0.12
        assert!((turnover_values.get(2).unwrap() - 0.12).abs() < 1e-6);
    }
}

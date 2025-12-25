//! Gross Profitability factor (Novy-Marx 2013).
//!
//! Gross profitability measures a company's gross profits scaled by total assets.
//! This quality factor has shown strong predictive power for future returns.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Gross Profitability factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct GrossProfitabilityConfig {}

/// Gross Profitability factor.
///
/// Gross profitability is calculated as:
/// ```text
/// Gross Profitability = Gross Profit / Total Assets
/// ```
///
/// where Gross Profit = Revenue - COGS (Cost of Goods Sold)
///
/// This factor was introduced by Novy-Marx (2013) and has been shown to be
/// a powerful predictor of cross-sectional stock returns. It captures the
/// efficiency of a firm's production process and is more stable than
/// earnings-based measures.
///
/// Reference: Novy-Marx, R. (2013). "The other side of value: The gross
/// profitability premium." Journal of Financial Economics, 108(1), 1-28.
#[derive(Debug, Clone, Default)]
pub struct GrossProfitability {
    config: GrossProfitabilityConfig,
}

impl Factor for GrossProfitability {
    fn name(&self) -> &str {
        "gross_profitability"
    }

    fn description(&self) -> &str {
        "Gross profitability (Novy-Marx) - gross profit scaled by total assets"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "revenue", "cogs", "total_assets"]
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
                ((col("revenue") - col("cogs")) / col("total_assets")).alias("gross_profitability"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for GrossProfitability {
    type Config = GrossProfitabilityConfig;

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
    fn test_gross_profitability_metadata() {
        let factor = GrossProfitability::default();
        assert_eq!(factor.name(), "gross_profitability");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_gross_profitability_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "revenue" => [100000.0, 75000.0, 80000.0],
            "cogs" => [40000.0, 30000.0, 20000.0],
            "total_assets" => [500000.0, 300000.0, 400000.0]
        ]
        .unwrap();

        let factor = GrossProfitability::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let gp_values = result.column("gross_profitability").unwrap().f64().unwrap();

        // AAPL: (100000 - 40000) / 500000 = 60000 / 500000 = 0.12
        assert!((gp_values.get(0).unwrap() - 0.12).abs() < 1e-6);

        // GOOGL: (75000 - 30000) / 300000 = 45000 / 300000 = 0.15
        assert!((gp_values.get(1).unwrap() - 0.15).abs() < 1e-6);

        // MSFT: (80000 - 20000) / 400000 = 60000 / 400000 = 0.15
        assert!((gp_values.get(2).unwrap() - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_gross_profitability_required_columns() {
        let factor = GrossProfitability::default();
        let required = factor.required_columns();
        assert_eq!(required.len(), 5);
        assert!(required.contains(&"symbol"));
        assert!(required.contains(&"date"));
        assert!(required.contains(&"revenue"));
        assert!(required.contains(&"cogs"));
        assert!(required.contains(&"total_assets"));
    }

    #[test]
    fn test_gross_profitability_edge_cases() {
        // Test with high gross profitability
        let df = df![
            "symbol" => ["HIGH_GP", "LOW_GP"],
            "date" => ["2024-03-31", "2024-03-31"],
            "revenue" => [100000.0, 100000.0],
            "cogs" => [10000.0, 90000.0],
            "total_assets" => [100000.0, 100000.0]
        ]
        .unwrap();

        let factor = GrossProfitability::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let gp_values = result.column("gross_profitability").unwrap().f64().unwrap();

        // HIGH_GP: (100000 - 10000) / 100000 = 0.90
        assert!((gp_values.get(0).unwrap() - 0.90).abs() < 1e-6);

        // LOW_GP: (100000 - 90000) / 100000 = 0.10
        assert!((gp_values.get(1).unwrap() - 0.10).abs() < 1e-6);
    }
}

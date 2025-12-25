//! Altman Z-Score factor.
//!
//! The Altman Z-Score is a bankruptcy prediction model that combines multiple
//! financial ratios to assess a company's financial health and bankruptcy risk.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Altman Z-Score factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct AltmanZConfig {}

/// Altman Z-Score factor.
///
/// The Z-Score is calculated as:
/// ```text
/// Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
/// where:
///   A = Working Capital / Total Assets
///   B = Retained Earnings / Total Assets
///   C = EBIT / Total Assets
///   D = Market Cap / Total Liabilities
///   E = Revenue / Total Assets
/// ```
///
/// Higher Z-scores indicate lower bankruptcy risk and higher financial quality.
/// - Z > 2.99: "Safe" zone
/// - 1.81 < Z < 2.99: "Grey" zone
/// - Z < 1.81: "Distress" zone
#[derive(Debug, Clone, Default)]
pub struct AltmanZ {
    config: AltmanZConfig,
}

impl Factor for AltmanZ {
    fn name(&self) -> &str {
        "altman_z_score"
    }

    fn description(&self) -> &str {
        "Altman Z-Score - bankruptcy prediction model combining multiple financial ratios"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &[
            "symbol",
            "date",
            "working_capital",
            "retained_earnings",
            "ebit",
            "market_cap",
            "total_liabilities",
            "revenue",
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
                (lit(1.2) * (col("working_capital") / col("total_assets"))
                    + lit(1.4) * (col("retained_earnings") / col("total_assets"))
                    + lit(3.3) * (col("ebit") / col("total_assets"))
                    + lit(0.6) * (col("market_cap") / col("total_liabilities"))
                    + lit(1.0) * (col("revenue") / col("total_assets")))
                .alias("altman_z_score"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for AltmanZ {
    type Config = AltmanZConfig;

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
    fn test_altman_z_metadata() {
        let factor = AltmanZ::default();
        assert_eq!(factor.name(), "altman_z_score");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
        assert_eq!(factor.required_columns().len(), 9);
    }

    #[test]
    fn test_altman_z_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "working_capital" => [50000.0, 40000.0, 60000.0],
            "retained_earnings" => [30000.0, 25000.0, 35000.0],
            "ebit" => [20000.0, 15000.0, 25000.0],
            "market_cap" => [300000.0, 250000.0, 350000.0],
            "total_liabilities" => [100000.0, 125000.0, 150000.0],
            "revenue" => [80000.0, 70000.0, 90000.0],
            "total_assets" => [200000.0, 175000.0, 225000.0]
        ]
        .unwrap();

        let factor = AltmanZ::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let z_scores = result.column("altman_z_score").unwrap().f64().unwrap();

        // AAPL: 1.2*(50000/200000) + 1.4*(30000/200000) + 3.3*(20000/200000) + 0.6*(300000/100000) + 1.0*(80000/200000)
        //     = 1.2*0.25 + 1.4*0.15 + 3.3*0.1 + 0.6*3.0 + 1.0*0.4
        //     = 0.3 + 0.21 + 0.33 + 1.8 + 0.4 = 3.04
        assert!((z_scores.get(0).unwrap() - 3.04).abs() < 1e-6);
    }

    #[test]
    fn test_altman_z_distress_zone() {
        // Create a company in financial distress (low Z-score)
        let df = df![
            "symbol" => ["WEAK"],
            "date" => ["2024-03-31"],
            "working_capital" => [-10000.0],  // Negative working capital
            "retained_earnings" => [5000.0],
            "ebit" => [2000.0],
            "market_cap" => [50000.0],
            "total_liabilities" => [150000.0],
            "revenue" => [30000.0],
            "total_assets" => [200000.0]
        ]
        .unwrap();

        let factor = AltmanZ::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let z_scores = result.column("altman_z_score").unwrap().f64().unwrap();

        // Should be in distress zone (< 1.81)
        assert!(z_scores.get(0).unwrap() < 1.81);
    }
}

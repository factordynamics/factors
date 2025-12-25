//! Return on Assets (ROA) factor.
//!
//! ROA measures a company's profitability relative to its total assets.
//! Higher ROA indicates more efficient asset utilization to generate profits.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Return on Assets factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoaConfig {}

/// Return on Assets factor.
///
/// ROA is calculated as:
/// ```text
/// ROA = Net Income / Total Assets
/// ```
///
/// This factor is used in quality-based strategies to identify companies
/// that efficiently convert their asset base into profits.
#[derive(Debug, Clone, Default)]
pub struct Roa {
    config: RoaConfig,
}

impl Factor for Roa {
    fn name(&self) -> &str {
        "roa"
    }

    fn description(&self) -> &str {
        "Return on Assets - net income divided by total assets"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "net_income", "total_assets"]
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
                (col("net_income") / col("total_assets")).alias("roa"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for Roa {
    type Config = RoaConfig;

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
    fn test_roa_metadata() {
        let factor = Roa::default();
        assert_eq!(factor.name(), "roa");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_roa_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "net_income" => [25000.0, 15000.0, 20000.0],
            "total_assets" => [500000.0, 300000.0, 400000.0]
        ]
        .unwrap();

        let factor = Roa::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let roa_values = result.column("roa").unwrap().f64().unwrap();
        assert!((roa_values.get(0).unwrap() - 0.05).abs() < 1e-6); // 25000/500000
        assert!((roa_values.get(1).unwrap() - 0.05).abs() < 1e-6); // 15000/300000
        assert!((roa_values.get(2).unwrap() - 0.05).abs() < 1e-6); // 20000/400000
    }
}

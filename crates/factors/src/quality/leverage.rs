//! Leverage factor.
//!
//! Leverage measures a company's financial leverage through its debt-to-equity ratio.
//! Higher leverage indicates greater financial risk from debt obligations.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Leverage factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeverageConfig {}

/// Leverage factor.
///
/// Leverage is calculated as:
/// ```text
/// Leverage = Total Debt / Shareholders' Equity
/// ```
///
/// This factor is used in quality-based strategies to assess financial risk.
/// Lower leverage generally indicates more conservative capital structure and
/// lower financial risk, though optimal leverage varies by industry.
#[derive(Debug, Clone, Default)]
pub struct Leverage {
    config: LeverageConfig,
}

impl Factor for Leverage {
    fn name(&self) -> &str {
        "leverage"
    }

    fn description(&self) -> &str {
        "Leverage - total debt divided by shareholders' equity"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "total_debt", "shareholders_equity"]
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
                (col("total_debt") / col("shareholders_equity")).alias("leverage"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for Leverage {
    type Config = LeverageConfig;

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
    fn test_leverage_metadata() {
        let factor = Leverage::default();
        assert_eq!(factor.name(), "leverage");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_leverage_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "total_debt" => [50000.0, 30000.0, 80000.0],
            "shareholders_equity" => [100000.0, 150000.0, 200000.0]
        ]
        .unwrap();

        let factor = Leverage::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let leverage_values = result.column("leverage").unwrap().f64().unwrap();
        assert!((leverage_values.get(0).unwrap() - 0.5).abs() < 1e-6); // 50000/100000
        assert!((leverage_values.get(1).unwrap() - 0.2).abs() < 1e-6); // 30000/150000
        assert!((leverage_values.get(2).unwrap() - 0.4).abs() < 1e-6); // 80000/200000
    }
}

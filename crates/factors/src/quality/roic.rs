//! Return on Invested Capital (ROIC) factor.
//!
//! ROIC measures how efficiently a company generates profits from its invested capital.
//! Higher ROIC suggests better capital allocation and operational efficiency.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Return on Invested Capital factor.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoicConfig {}

/// Return on Invested Capital factor.
///
/// ROIC is calculated as:
/// ```text
/// ROIC = NOPAT / Invested Capital
/// where NOPAT = Operating Income * (1 - tax_rate)
/// ```
///
/// This factor measures how well a company generates profits from its invested capital,
/// providing insight into capital allocation efficiency.
#[derive(Debug, Clone, Default)]
pub struct Roic {
    config: RoicConfig,
}

impl Factor for Roic {
    fn name(&self) -> &str {
        "roic"
    }

    fn description(&self) -> &str {
        "Return on Invested Capital - NOPAT divided by invested capital"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &[
            "symbol",
            "date",
            "operating_income",
            "tax_rate",
            "invested_capital",
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
                ((col("operating_income") * (lit(1.0) - col("tax_rate")))
                    / col("invested_capital"))
                .alias("roic"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for Roic {
    type Config = RoicConfig;

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
    fn test_roic_metadata() {
        let factor = Roic::default();
        assert_eq!(factor.name(), "roic");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_roic_computation() {
        let df = df![
            "symbol" => ["AAPL", "GOOGL", "MSFT"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "operating_income" => [30000.0, 20000.0, 25000.0],
            "tax_rate" => [0.2, 0.25, 0.21],
            "invested_capital" => [100000.0, 80000.0, 125000.0]
        ]
        .unwrap();

        let factor = Roic::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (3, 3));

        let roic_values = result.column("roic").unwrap().f64().unwrap();
        // AAPL: 30000 * (1 - 0.2) / 100000 = 24000 / 100000 = 0.24
        assert!((roic_values.get(0).unwrap() - 0.24).abs() < 1e-6);
        // GOOGL: 20000 * (1 - 0.25) / 80000 = 15000 / 80000 = 0.1875
        assert!((roic_values.get(1).unwrap() - 0.1875).abs() < 1e-6);
        // MSFT: 25000 * (1 - 0.21) / 125000 = 19750 / 125000 = 0.158
        assert!((roic_values.get(2).unwrap() - 0.158).abs() < 1e-6);
    }
}

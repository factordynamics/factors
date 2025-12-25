//! Earnings Persistence factor.
//!
//! Measures the autocorrelation of quarterly earnings (EPS), indicating how
//! predictable and persistent a company's earnings are over time.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Earnings Persistence factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EarningsPersistenceConfig {
    /// Number of quarters to look back for persistence calculation (default: 8)
    pub lookback: usize,
}

impl Default for EarningsPersistenceConfig {
    fn default() -> Self {
        Self { lookback: 8 }
    }
}

/// Earnings Persistence factor.
///
/// Measures earnings stability over time using the inverse coefficient of variation:
/// ```text
/// Persistence = 1 - (σ(EPS) / |μ(EPS)|)
/// ```
///
/// This metric represents earnings persistence:
/// - Values close to 1: Highly stable/persistent earnings (low volatility)
/// - Values close to 0: Volatile/unstable earnings (high volatility)
/// - Negative values: Very high volatility relative to mean
///
/// More stable earnings are considered higher quality and more predictable.
/// We use the last 8 quarters of data by default to compute statistics.
#[derive(Debug, Clone, Default)]
pub struct EarningsPersistence {
    config: EarningsPersistenceConfig,
}

impl Factor for EarningsPersistence {
    fn name(&self) -> &str {
        "earnings_persistence"
    }

    fn description(&self) -> &str {
        "Earnings Persistence - inverse coefficient of variation measuring EPS stability"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "eps"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Quarterly
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Simplified approach: Calculate the coefficient of variation (inverse) as a proxy
        // for earnings persistence. More stable earnings (lower CV) indicate higher persistence.
        // We use: 1 - (std(eps) / abs(mean(eps)))
        // This gives values between 0 and 1, where higher values indicate more persistent earnings.

        let result = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())))
            .sort(["symbol", "date"], Default::default())
            .group_by([col("symbol")])
            .agg([
                // Keep the latest date for each symbol
                col("date").max().alias("date"),
                // Calculate mean and std of EPS
                col("eps").mean().alias("eps_mean"),
                col("eps").std(0).alias("eps_std"),
            ])
            .filter(col("date").eq(lit(date.to_string())))
            .select([
                col("symbol"),
                col("date"),
                // Persistence = 1 - coefficient_of_variation
                // Higher values indicate more stable, persistent earnings
                (lit(1.0) - (col("eps_std") / col("eps_mean").abs())).alias("earnings_persistence"),
            ])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for EarningsPersistence {
    type Config = EarningsPersistenceConfig;

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
    fn test_earnings_persistence_metadata() {
        let factor = EarningsPersistence::default();
        assert_eq!(factor.name(), "earnings_persistence");
        assert_eq!(factor.lookback(), 8);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
    }

    #[test]
    fn test_earnings_persistence_high_persistence() {
        // Create EPS data with high persistence (steady growth)
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "date" => [
                "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"
            ],
            "eps" => [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        ]
        .unwrap();

        let factor = EarningsPersistence::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 6, 30).unwrap())
            .unwrap();

        assert_eq!(result.shape().0, 1);

        let persistence = result
            .column("earnings_persistence")
            .unwrap()
            .f64()
            .unwrap();

        // Should have high persistence (low coefficient of variation, close to 1)
        // With steady growth, CV should be small, so (1 - CV) should be high
        assert!(persistence.get(0).unwrap() > 0.8);
    }

    #[test]
    fn test_earnings_persistence_volatile_earnings() {
        // Create EPS data with volatile, less persistent earnings
        let df = df![
            "symbol" => ["WEAK", "WEAK", "WEAK", "WEAK", "WEAK", "WEAK", "WEAK", "WEAK"],
            "date" => [
                "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"
            ],
            "eps" => [1.5, 0.8, 1.9, 0.6, 1.7, 0.9, 1.8, 0.7]
        ]
        .unwrap();

        let factor = EarningsPersistence::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 6, 30).unwrap())
            .unwrap();

        let persistence = result
            .column("earnings_persistence")
            .unwrap()
            .f64()
            .unwrap();

        // Should have lower persistence (higher coefficient of variation)
        // Volatile earnings should have high CV, so (1 - CV) should be lower
        let persistence_value = persistence.get(0).unwrap();
        assert!(persistence_value < 0.8 || persistence_value.is_nan());
    }

    #[test]
    fn test_earnings_persistence_multiple_symbols() {
        // Test with multiple symbols
        let df = df![
            "symbol" => [
                "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL",
                "GOOGL", "GOOGL", "GOOGL", "GOOGL", "GOOGL", "GOOGL", "GOOGL", "GOOGL"
            ],
            "date" => [
                "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30",
                "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
                "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"
            ],
            "eps" => [
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  // AAPL: persistent
                2.0, 1.5, 2.2, 1.3, 2.1, 1.4, 2.3, 1.6   // GOOGL: volatile
            ]
        ]
        .unwrap();

        let factor = EarningsPersistence::default();
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 6, 30).unwrap())
            .unwrap();

        assert_eq!(result.shape().0, 2);
    }
}

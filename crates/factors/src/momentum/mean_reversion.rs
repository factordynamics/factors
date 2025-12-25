//! Mean Reversion / Bollinger Band factor.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for mean reversion factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeanReversionConfig {
    /// Lookback window in days for SMA and std dev (default: 20)
    pub lookback: usize,
    /// Number of standard deviations for Bollinger Bands (default: 2.0)
    pub num_std_dev: f64,
}

impl Default for MeanReversionConfig {
    fn default() -> Self {
        Self {
            lookback: 20,
            num_std_dev: 2.0,
        }
    }
}

/// Mean reversion factor measuring price distance from moving average.
///
/// Calculates how far the current price has deviated from its moving average,
/// normalized by volatility:
/// `(P_t - SMA) / (num_std_dev * σ)`
///
/// where:
/// - `P_t` is the current price
/// - `SMA` is the simple moving average over lookback period (default: 20-day)
/// - `σ` is the standard deviation over lookback period
/// - `num_std_dev` is the number of standard deviations (default: 2.0)
///
/// This is essentially the Bollinger Band position:
/// - Values near +1.0 indicate price at upper band (potentially overbought)
/// - Values near -1.0 indicate price at lower band (potentially oversold)
/// - Values near 0 indicate price near the mean
///
/// Captures mean reversion tendencies and identifies extreme price moves.
#[derive(Debug, Clone, Default)]
pub struct MeanReversion {
    config: MeanReversionConfig,
}

impl Factor for MeanReversion {
    fn name(&self) -> &str {
        "mean_reversion"
    }

    fn description(&self) -> &str {
        "20-day mean reversion - price distance from moving average in standard deviations"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Filter data up to the target date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.format("%Y-%m-%d").to_string())))
            .collect()?;

        let lookback = self.config.lookback;
        let num_std_dev = self.config.num_std_dev;

        // Group by symbol and compute mean reversion
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                // Get current price
                col("close")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("current_price"),
                // Get last N prices for SMA and std dev calculation
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(lookback))
                    .alias("prices"),
            ])
            .with_columns([
                // Calculate moving average
                col("prices").list().mean().alias("sma"),
                // Calculate standard deviation
                col("prices").list().std(1).alias("std"),
            ])
            .with_column(
                // (P_t - SMA) / (num_std_dev * σ)
                ((col("current_price") - col("sma")) / (lit(num_std_dev) * col("std")))
                    .alias(self.name()),
            )
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for MeanReversion {
    type Config = MeanReversionConfig;

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
    use polars::df;

    #[test]
    fn test_mean_reversion_at_mean() {
        let factor = MeanReversion::default();

        // Create test data with stable prices around 100
        let dates: Vec<String> = (0..20)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 20];
        // Prices oscillating around 100 with small variance
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + (i % 3) as f64 - 1.0).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 20).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Mean reversion should be near 0 when price is near average
        let mean_rev = result
            .column("mean_reversion")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            mean_rev.abs() < 1.0,
            "Expected mean reversion near 0, got {}",
            mean_rev
        );
    }

    #[test]
    fn test_mean_reversion_extreme_high() {
        let factor = MeanReversion::default();

        // Create test data where last price is much higher than average
        let dates: Vec<String> = (0..21)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["MSFT"; 21];
        // First 20 prices stable around 100, last price jumps to 120
        let mut prices: Vec<f64> = (0..20).map(|_| 100.0).collect();
        prices.push(120.0);

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 21).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // Mean reversion should be strongly positive (price above upper band)
        let mean_rev = result
            .column("mean_reversion")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            mean_rev > 1.0,
            "Expected positive mean reversion for high price, got {}",
            mean_rev
        );
    }

    #[test]
    fn test_mean_reversion_metadata() {
        let factor = MeanReversion::default();

        assert_eq!(factor.name(), "mean_reversion");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

//! Relative Strength Index (RSI) factor.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for RSI factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RSIConfig {
    /// Lookback period in days (default: 14)
    pub period: usize,
}

impl Default for RSIConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

/// RSI factor measuring momentum strength on a 0-100 scale.
///
/// The Relative Strength Index (RSI) is calculated as:
/// `RSI = 100 - (100 / (1 + RS))`
///
/// where:
/// - `RS = avg_gain / avg_loss` over the period
/// - `avg_gain` is the average of price increases over the period
/// - `avg_loss` is the average of price decreases over the period
///
/// RSI values range from 0 to 100:
/// - Above 70 typically indicates overbought conditions
/// - Below 30 typically indicates oversold conditions
/// - 50 represents neutral momentum
///
/// Captures short-term momentum and potential reversal points.
#[derive(Debug, Clone, Default)]
pub struct RSI {
    config: RSIConfig,
}

impl Factor for RSI {
    fn name(&self) -> &str {
        "rsi"
    }

    fn description(&self) -> &str {
        "14-day Relative Strength Index - momentum strength indicator"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        self.config.period
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

        // Calculate RSI for each symbol by processing grouped data
        let mut results = Vec::new();

        let grouped = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last(),
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(15)),
            ])
            .collect()?;

        for idx in 0..grouped.height() {
            let symbol = grouped.column("symbol")?.get(idx)?;
            let date_val = grouped.column("date")?.get(idx)?;
            let prices = grouped
                .column("close")?
                .list()?
                .get_as_series(idx)
                .ok_or_else(|| {
                    crate::FactorError::Computation("Failed to get prices series".to_string())
                })?;

            if prices.len() < 15 {
                continue;
            }

            let price_vec: Vec<f64> = prices.f64()?.to_vec().into_iter().flatten().collect();

            // Calculate price changes
            let mut gains = Vec::new();
            let mut losses = Vec::new();

            for i in 1..price_vec.len() {
                let change = price_vec[i] - price_vec[i - 1];
                if change > 0.0 {
                    gains.push(change);
                    losses.push(0.0);
                } else {
                    gains.push(0.0);
                    losses.push(-change);
                }
            }

            let avg_gain: f64 = gains.iter().sum::<f64>() / gains.len() as f64;
            let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;

            let rsi = if avg_loss == 0.0 {
                100.0
            } else {
                let rs = avg_gain / avg_loss;
                100.0 - (100.0 / (1.0 + rs))
            };

            results.push((symbol, date_val, rsi));
        }

        // Build result DataFrame
        let symbols: Vec<_> = results.iter().map(|(s, _, _)| s.clone()).collect();
        let dates: Vec<_> = results.iter().map(|(_, d, _)| d.clone()).collect();
        let rsi_values: Vec<f64> = results.iter().map(|(_, _, r)| *r).collect();

        let df = DataFrame::new(vec![
            Series::new("symbol".into(), symbols).into(),
            Series::new("date".into(), dates).into(),
            Series::new(self.name().into(), rsi_values).into(),
        ])?;

        Ok(df)
    }
}

impl ConfigurableFactor for RSI {
    type Config = RSIConfig;

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
    fn test_rsi_trending_up() {
        let factor = RSI::default();

        // Create test data with 15 days of consistently rising prices
        let dates: Vec<String> = (0..15)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 15];
        // Prices consistently rising
        let prices: Vec<f64> = (0..15).map(|i| 100.0 + i as f64 * 2.0).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 15).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // RSI should be high (near 100) for consistently rising prices
        let rsi = result.column("rsi").unwrap().f64().unwrap().get(0).unwrap();

        assert!(
            rsi > 90.0,
            "Expected RSI > 90 for rising prices, got {}",
            rsi
        );
    }

    #[test]
    fn test_rsi_mixed_movements() {
        let factor = RSI::default();

        // Create test data with mixed price movements
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

        let symbols = vec!["MSFT"; 20];
        // Oscillating prices: up, down, up, down...
        let prices: Vec<f64> = (0..20)
            .map(|i| 100.0 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

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

        // RSI should be near 50 for balanced movements
        let rsi = result.column("rsi").unwrap().f64().unwrap().get(0).unwrap();

        assert!(
            rsi > 40.0 && rsi < 60.0,
            "Expected RSI near 50 for mixed movements, got {}",
            rsi
        );
    }

    #[test]
    fn test_rsi_metadata() {
        let factor = RSI::default();

        assert_eq!(factor.name(), "rsi");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 14);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

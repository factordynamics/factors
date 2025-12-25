//! Volatility Breakout factor.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for volatility breakout factor.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct VolatilityBreakoutConfig {
    /// Number of trading days for SMA and ATR calculation (default: 20)
    pub lookback: usize,
}

impl Default for VolatilityBreakoutConfig {
    fn default() -> Self {
        Self { lookback: 20 }
    }
}

/// Volatility Breakout factor measuring price moves relative to volatility.
///
/// Calculates how far the current price has moved from its 20-day average,
/// scaled by the Average True Range (ATR):
/// `(P_t - SMA_20) / ATR_20`
///
/// where:
/// - `P_t` is the current close price
/// - `SMA_20` is the 20-day simple moving average of close prices
/// - `ATR_20` is the 20-day Average True Range
/// - `TR = max(high - low, |high - prev_close|, |low - prev_close|)`
///
/// This indicator identifies significant price breakouts:
/// - Positive values indicate upward breakout (price above average)
/// - Negative values indicate downward breakout (price below average)
/// - Magnitude shows breakout strength relative to typical volatility
///
/// Normalizing by ATR makes breakouts comparable across different securities
/// and volatility regimes.
#[derive(Debug, Clone, Default)]
pub struct VolatilityBreakout {
    config: VolatilityBreakoutConfig,
}

impl Factor for VolatilityBreakout {
    fn name(&self) -> &str {
        "volatility_breakout"
    }

    fn description(&self) -> &str {
        "20-day volatility breakout - price deviation scaled by ATR"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "high", "low", "close"]
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

        // Calculate volatility breakout for each symbol by processing grouped data
        let mut results = Vec::new();

        let grouped = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last(),
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(self.lookback() + 1)),
                col("high")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(self.lookback())),
                col("low")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(self.lookback())),
            ])
            .collect()?;

        for idx in 0..grouped.height() {
            let symbol = grouped.column("symbol")?.get(idx)?;
            let date_val = grouped.column("date")?.get(idx)?;
            let closes = grouped
                .column("close")?
                .list()?
                .get_as_series(idx)
                .ok_or_else(|| {
                    crate::FactorError::Computation("Failed to get closes series".to_string())
                })?;
            let highs = grouped
                .column("high")?
                .list()?
                .get_as_series(idx)
                .ok_or_else(|| {
                    crate::FactorError::Computation("Failed to get highs series".to_string())
                })?;
            let lows = grouped
                .column("low")?
                .list()?
                .get_as_series(idx)
                .ok_or_else(|| {
                    crate::FactorError::Computation("Failed to get lows series".to_string())
                })?;

            if closes.len() < (self.lookback() + 1)
                || highs.len() < self.lookback()
                || lows.len() < self.lookback()
            {
                continue;
            }

            let close_vec: Vec<f64> = closes.f64()?.to_vec().into_iter().flatten().collect();
            let high_vec: Vec<f64> = highs.f64()?.to_vec().into_iter().flatten().collect();
            let low_vec: Vec<f64> = lows.f64()?.to_vec().into_iter().flatten().collect();

            // Calculate current price
            let current_price = close_vec[self.lookback()];

            // Calculate SMA_20 from last 20 closes
            let sma_20: f64 =
                close_vec[1..=self.lookback()].iter().sum::<f64>() / self.lookback() as f64;

            // Calculate ATR_20
            let mut true_ranges = Vec::new();
            for i in 0..self.lookback() {
                let high = high_vec[i];
                let low = low_vec[i];
                let prev_close = close_vec[i];

                let hl_range = high - low;
                let hc_range = (high - prev_close).abs();
                let lc_range = (low - prev_close).abs();

                let true_range = hl_range.max(hc_range).max(lc_range);
                true_ranges.push(true_range);
            }

            let atr_20 = true_ranges.iter().sum::<f64>() / true_ranges.len() as f64;

            // Calculate breakout: (P_t - SMA_20) / ATR_20
            let breakout = if atr_20 > 0.0 {
                (current_price - sma_20) / atr_20
            } else {
                0.0 // Avoid division by zero
            };

            results.push((symbol, date_val, breakout));
        }

        // Build result DataFrame
        let symbols: Vec<_> = results.iter().map(|(s, _, _)| s.clone()).collect();
        let dates: Vec<_> = results.iter().map(|(_, d, _)| d.clone()).collect();
        let breakout_values: Vec<f64> = results.iter().map(|(_, _, b)| *b).collect();

        let df = DataFrame::new(vec![
            Series::new("symbol".into(), symbols).into(),
            Series::new("date".into(), dates).into(),
            Series::new(self.name().into(), breakout_values).into(),
        ])?;

        Ok(df)
    }
}

impl ConfigurableFactor for VolatilityBreakout {
    type Config = VolatilityBreakoutConfig;

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
    fn test_volatility_breakout_upward() {
        let factor = VolatilityBreakout::default();

        // Create test data with stable prices then a breakout
        let dates: Vec<String> = (0..22)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 22];
        // First 20 days stable around 100, last day jumps to 110
        let mut closes: Vec<f64> = (0..21).map(|_| 100.0).collect();
        closes.push(110.0);
        // Highs slightly above close
        let highs: Vec<f64> = closes.iter().map(|c| c + 1.0).collect();
        // Lows slightly below close
        let lows: Vec<f64> = closes.iter().map(|c| c - 1.0).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "high" => highs,
            "low" => lows,
            "close" => closes,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 22).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Breakout should be strongly positive
        let breakout = result
            .column("volatility_breakout")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            breakout > 2.0,
            "Expected strong positive breakout, got {}",
            breakout
        );
    }

    #[test]
    fn test_volatility_breakout_near_average() {
        let factor = VolatilityBreakout::default();

        // Create test data with prices oscillating around mean
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
        // Prices oscillating around 100 with small range
        let closes: Vec<f64> = (0..21).map(|i| 100.0 + (i % 3) as f64 - 1.0).collect();
        let highs: Vec<f64> = closes.iter().map(|c| c + 1.0).collect();
        let lows: Vec<f64> = closes.iter().map(|c| c - 1.0).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "high" => highs,
            "low" => lows,
            "close" => closes,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 21).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // Breakout should be near 0 when price is near average
        let breakout = result
            .column("volatility_breakout")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            breakout.abs() < 1.0,
            "Expected breakout near 0, got {}",
            breakout
        );
    }

    #[test]
    fn test_volatility_breakout_metadata() {
        let factor = VolatilityBreakout::default();

        assert_eq!(factor.name(), "volatility_breakout");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "high", "low", "close"]
        );
    }
}

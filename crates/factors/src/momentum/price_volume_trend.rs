//! Price-Volume Trend (PVT) factor.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for Price-Volume Trend factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PriceVolumeTrendConfig {
    /// Lookback window in days (default: 20)
    pub lookback: usize,
}

impl Default for PriceVolumeTrendConfig {
    fn default() -> Self {
        Self { lookback: 20 }
    }
}

/// Price-Volume Trend factor combining price momentum with volume.
///
/// Calculates the cumulative sum of volume-weighted price changes:
/// `PVT = Σ((P_t - P_{t-1})/P_{t-1}) * V_t` over the lookback period
///
/// where:
/// - `P_t` is the current day's close price
/// - `P_{t-1}` is the previous day's close price
/// - `V_t` is the current day's volume
///
/// This indicator combines price direction with volume strength:
/// - Rising PVT indicates accumulation (bullish with volume confirmation)
/// - Falling PVT indicates distribution (bearish with volume confirmation)
/// - Divergences between PVT and price can signal reversals
///
/// Captures momentum quality by weighting price changes by trading volume.
#[derive(Debug, Clone, Default)]
pub struct PriceVolumeTrend {
    config: PriceVolumeTrendConfig,
}

impl Factor for PriceVolumeTrend {
    fn name(&self) -> &str {
        "price_volume_trend"
    }

    fn description(&self) -> &str {
        "20-day price-volume trend - volume-weighted cumulative momentum"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close", "volume"]
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

        // Calculate PVT for each symbol by processing grouped data
        let mut results = Vec::new();

        let grouped = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last(),
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(lookback + 1)),
                col("volume")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(lookback)),
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
            let volumes = grouped
                .column("volume")?
                .list()?
                .get_as_series(idx)
                .ok_or_else(|| {
                    crate::FactorError::Computation("Failed to get volumes series".to_string())
                })?;

            if prices.len() < (lookback + 1) || volumes.len() < lookback {
                continue;
            }

            let price_vec: Vec<f64> = prices.f64()?.to_vec().into_iter().flatten().collect();
            let volume_vec: Vec<f64> = volumes.f64()?.to_vec().into_iter().flatten().collect();

            // Calculate volume-weighted price changes
            let mut pvt = 0.0;
            for i in 0..lookback {
                let price_change = (price_vec[i + 1] - price_vec[i]) / price_vec[i];
                pvt += price_change * volume_vec[i];
            }

            results.push((symbol, date_val, pvt));
        }

        // Build result DataFrame
        let symbols: Vec<_> = results.iter().map(|(s, _, _)| s.clone()).collect();
        let dates: Vec<_> = results.iter().map(|(_, d, _)| d.clone()).collect();
        let pvt_values: Vec<f64> = results.iter().map(|(_, _, p)| *p).collect();

        let df = DataFrame::new(vec![
            Series::new("symbol".into(), symbols).into(),
            Series::new("date".into(), dates).into(),
            Series::new(self.name().into(), pvt_values).into(),
        ])?;

        Ok(df)
    }
}

impl ConfigurableFactor for PriceVolumeTrend {
    type Config = PriceVolumeTrendConfig;

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
    fn test_pvt_rising_prices_with_volume() {
        let factor = PriceVolumeTrend::default();

        // Create test data with rising prices and consistent volume
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

        let symbols = vec!["AAPL"; 21];
        // Prices rising from 100 to 110
        let prices: Vec<f64> = (0..21).map(|i| 100.0 + i as f64 * 0.5).collect();
        // Constant volume
        let volumes: Vec<f64> = vec![1000000.0; 21];

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
            "volume" => volumes,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 21).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // PVT should be positive for rising prices
        let pvt = result
            .column("price_volume_trend")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            pvt > 0.0,
            "Expected positive PVT for rising prices, got {}",
            pvt
        );
    }

    #[test]
    fn test_pvt_falling_prices_with_volume() {
        let factor = PriceVolumeTrend::default();

        // Create test data with falling prices and consistent volume
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
        // Prices falling from 100 to 90
        let prices: Vec<f64> = (0..21).map(|i| 100.0 - i as f64 * 0.5).collect();
        // Constant volume
        let volumes: Vec<f64> = vec![1000000.0; 21];

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
            "volume" => volumes,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 21).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // PVT should be negative for falling prices
        let pvt = result
            .column("price_volume_trend")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            pvt < 0.0,
            "Expected negative PVT for falling prices, got {}",
            pvt
        );
    }

    #[test]
    fn test_pvt_metadata() {
        let factor = PriceVolumeTrend::default();

        assert_eq!(factor.name(), "price_volume_trend");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "close", "volume"]
        );
    }
}

//! 52-week high factor - behavioral anchoring momentum signal.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Configuration for 52-week high factor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct High52WeekConfig {
    /// Lookback window in days (default: 252 = 52 weeks)
    pub lookback: usize,
}

impl Default for High52WeekConfig {
    fn default() -> Self {
        Self { lookback: 252 }
    }
}

/// 52-week high factor measuring proximity to 52-week high.
///
/// Measures how close a stock's current price is to its high over the lookback period:
/// `P_t / max(P_{t-lookback:t})`
///
/// where:
/// - `P_t` is the current price
/// - `max(P_{t-lookback:t})` is the maximum price over the lookback period (default: 252 trading days)
///
/// This factor captures behavioral anchoring effects where stocks trading near
/// their highs tend to have momentum. Research shows that stocks near their
/// 52-week highs tend to continue outperforming, likely due to:
/// - Investor anchoring to recent price extremes
/// - Breakout effects as prices exceed psychological barriers
/// - Herding behavior as investors chase high-flying stocks
///
/// A value near 1.0 indicates the stock is at or near its high,
/// while lower values indicate the stock is trading below its peak.
#[derive(Debug, Clone, Default)]
pub struct High52Week {
    config: High52WeekConfig,
}

impl Factor for High52Week {
    fn name(&self) -> &str {
        "high_52week"
    }

    fn description(&self) -> &str {
        "52-week high ratio - behavioral anchoring momentum signal"
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

        // Group by symbol and compute 52-week high ratio
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                col("close")
                    .sort_by([col("date")], Default::default())
                    .last()
                    .alias("current_price"),
                col("close")
                    .sort_by([col("date")], Default::default())
                    .tail(Some(self.config.lookback + 1))
                    .max()
                    .alias("max_price"),
            ])
            .with_column((col("current_price") / col("max_price")).alias(self.name()))
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for High52Week {
    type Config = High52WeekConfig;

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
    fn test_high_52week_at_peak() {
        let factor = High52Week::default();

        // Create test data with 253 days of prices
        // Price steadily increases, so the last price is the 52-week high
        let dates: Vec<String> = (0..253)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 253];
        // Price goes from 100 to 150
        let prices: Vec<f64> = (0..253).map(|i| 100.0 + i as f64 * 0.1984).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 9, 10).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that the ratio is 1.0 (at the high)
        let ratio = result
            .column("high_52week")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!((ratio - 1.0).abs() < 0.001, "Expected ~1.0, got {}", ratio);
    }

    #[test]
    fn test_high_52week_below_peak() {
        let factor = High52Week::default();

        // Create test data where price peaks in the middle and then declines
        let dates: Vec<String> = (0..253)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["MSFT"; 253];
        // Price goes up to 150 at day 126, then back down to 120
        let prices: Vec<f64> = (0..253)
            .map(|i| {
                if i <= 126 {
                    100.0 + i as f64 * 0.3968
                } else {
                    150.0 - (i - 126) as f64 * 0.2362
                }
            })
            .collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 9, 10).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // Current price should be around 120, max should be around 150
        // So ratio should be around 0.8
        let ratio = result
            .column("high_52week")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!((ratio - 0.8).abs() < 0.05, "Expected ~0.8, got {}", ratio);
        assert!(ratio < 1.0, "Ratio should be less than 1.0");
    }

    #[test]
    fn test_high_52week_multiple_symbols() {
        let factor = High52Week::default();

        // Create test data for two symbols
        let mut dates = Vec::new();
        let mut symbols = Vec::new();
        let mut prices = Vec::new();

        // AAPL: at its high
        for i in 0..253 {
            dates.push(
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string(),
            );
            symbols.push("AAPL");
            prices.push(100.0 + i as f64 * 0.1984);
        }

        // MSFT: below its high
        for i in 0..253 {
            dates.push(
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string(),
            );
            symbols.push("MSFT");
            if i <= 126 {
                prices.push(100.0 + i as f64 * 0.3968);
            } else {
                prices.push(150.0 - (i - 126) as f64 * 0.2362);
            }
        }

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 9, 10).unwrap())
            .unwrap();

        assert_eq!(result.height(), 2);
        assert_eq!(result.width(), 3);

        // Check that we have both symbols
        let result_symbols = result
            .column("symbol")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>();

        assert!(result_symbols.contains(&"AAPL"));
        assert!(result_symbols.contains(&"MSFT"));
    }

    #[test]
    fn test_high_52week_metadata() {
        let factor = High52Week::default();

        assert_eq!(factor.name(), "high_52week");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

//! Short-Term Reversal factor - 1-week mean reversion signal.
//!
//! This factor captures short-term overreaction in stock prices. Stocks that
//! have declined over the past week tend to rebound, while recent winners
//! tend to reverse. This contrarian signal is distinct from longer-term
//! momentum effects.
//!
//! # Academic Foundation
//! Jegadeesh (1990) - "Evidence of Predictable Behavior of Security Returns"
//! Documents that short-term returns (1 week to 1 month) exhibit reversal
//! patterns due to market overreaction and liquidity-induced price movements.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Short-Term Reversal factor based on 1-week returns.
///
/// Computes the negative of the past week's return:
/// `-1 * (P_t / P_{t-5} - 1)`
///
/// where:
/// - `P_t` is the current price (close)
/// - `P_{t-5}` is the price 5 trading days ago
///
/// Negative multiplier converts the signal into a reversal signal:
/// - Positive values indicate stocks that declined recently (contrarian buy)
/// - Negative values indicate stocks that rose recently (contrarian sell)
///
/// This factor works best for highly liquid stocks where short-term price
/// movements may be driven by temporary order imbalances rather than
/// fundamental information.
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Trading date
/// - `close`: Closing price
///
/// # Lookback Period
/// 5 trading days (approximately 1 week)
#[derive(Debug, Clone, Default)]
pub struct ShortTermReversal;

impl Factor for ShortTermReversal {
    fn name(&self) -> &str {
        "short_term_reversal"
    }

    fn description(&self) -> &str {
        "Short-term reversal signal based on 1-week returns (Jegadeesh 1990)"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Sentiment
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        5 // 1 week of trading days
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

        // Group by symbol and compute reversal signal
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
                    .slice(
                        (lit(0) - lit(self.lookback() as i64 + 1)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("lagged_price"),
            ])
            .filter(col("lagged_price").gt(lit(0.0))) // Filter out invalid prices before calculation
            .filter(col("lagged_price").is_not_null()) // Filter out missing lagged prices
            .with_column(
                // Calculate reversal: -1 * (current / lagged - 1) = -1 * return
                (lit(-1.0) * ((col("current_price") / col("lagged_price")) - lit(1.0)))
                    .alias(self.name()),
            )
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    #[test]
    fn test_short_term_reversal_decline() {
        let factor = ShortTermReversal;

        // Create test data where price declined from $100 to $95 (-5%)
        let dates: Vec<String> = (0..6)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 6];
        let prices = vec![100.0, 99.0, 98.0, 97.0, 96.0, 95.0];

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 6).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Reversal = -1 * (95/100 - 1) = -1 * (-0.05) = 0.05 (positive, buy signal)
        let reversal = result
            .column("short_term_reversal")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (reversal - 0.05).abs() < 0.001,
            "Expected reversal of 0.05, got {}",
            reversal
        );
    }

    #[test]
    fn test_short_term_reversal_increase() {
        let factor = ShortTermReversal;

        // Create test data where price increased from $100 to $105 (+5%)
        let dates: Vec<String> = (0..6)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["MSFT"; 6];
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 6).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // Reversal = -1 * (105/100 - 1) = -1 * 0.05 = -0.05 (negative, sell signal)
        let reversal = result
            .column("short_term_reversal")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (reversal + 0.05).abs() < 0.001,
            "Expected reversal of -0.05, got {}",
            reversal
        );
    }

    #[test]
    fn test_short_term_reversal_multiple_stocks() {
        let factor = ShortTermReversal;

        // Create test data for multiple stocks with different patterns
        let mut dates = Vec::new();
        let mut symbols = Vec::new();
        let mut prices = Vec::new();

        // AAPL: declining from 100 to 95
        for i in 0..6 {
            dates.push(
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string(),
            );
            symbols.push("AAPL");
            prices.push(100.0 - i as f64);
        }

        // MSFT: increasing from 100 to 105
        for i in 0..6 {
            dates.push(
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string(),
            );
            symbols.push("MSFT");
            prices.push(100.0 + i as f64);
        }

        // GOOGL: flat at 100
        for i in 0..6 {
            dates.push(
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string(),
            );
            symbols.push("GOOGL");
            prices.push(100.0);
        }

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 6).unwrap())
            .unwrap();

        assert_eq!(result.height(), 3);
        assert!(result.column("short_term_reversal").is_ok());

        let syms = result.column("symbol").unwrap().str().unwrap();
        let reversals = result.column("short_term_reversal").unwrap().f64().unwrap();

        for i in 0..result.height() {
            let sym = syms.get(i).unwrap();
            let rev = reversals.get(i).unwrap();

            match sym {
                "AAPL" => assert!(
                    rev > 0.04,
                    "AAPL should have positive reversal, got {}",
                    rev
                ),
                "MSFT" => assert!(
                    rev < -0.04,
                    "MSFT should have negative reversal, got {}",
                    rev
                ),
                "GOOGL" => assert!(
                    rev.abs() < 0.01,
                    "GOOGL should have ~0 reversal, got {}",
                    rev
                ),
                _ => panic!("Unexpected symbol: {}", sym),
            }
        }
    }

    #[test]
    fn test_short_term_reversal_metadata() {
        let factor = ShortTermReversal;

        assert_eq!(factor.name(), "short_term_reversal");
        assert_eq!(factor.category(), FactorCategory::Sentiment);
        assert_eq!(factor.lookback(), 5);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }

    #[test]
    fn test_short_term_reversal_insufficient_history() {
        let factor = ShortTermReversal;

        // Create test data with only 3 days (insufficient for 5-day lookback)
        let dates: Vec<String> = (0..3)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 3];
        let prices = vec![100.0, 99.0, 98.0];

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 3).unwrap())
            .unwrap();

        // Should return empty due to insufficient history
        assert_eq!(result.height(), 0);
    }
}

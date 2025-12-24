//! Long-term momentum factor (12-month lookback with 1-month skip).

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Long-term momentum factor measuring 12-month price change with 1-month skip.
///
/// Measures the percentage price change from 12 months ago to 1 month ago:
/// `(P_{t-21} / P_{t-252}) - 1`
///
/// where:
/// - `P_{t-21}` is the price 21 trading days ago (1 month skip)
/// - `P_{t-252}` is the price 252 trading days ago (approximately 12 months)
///
/// The 1-month skip helps avoid short-term reversal effects while capturing
/// long-term trend persistence. This is based on the classic Jegadeesh-Titman
/// momentum strategy (1993).
///
/// Captures long-term trend persistence, useful for:
/// - Classic momentum strategies
/// - Annual rebalancing portfolios
/// - Identifying sustained multi-month trends
/// - Avoiding short-term mean reversion that often occurs in the most recent month
///
/// The skip period is crucial because stocks with very recent strong performance
/// often experience short-term reversals, while those with strong 12-month
/// performance (excluding the most recent month) tend to continue performing well.
#[derive(Debug, Clone, Default)]
pub struct LongTermMomentum;

impl LongTermMomentum {
    /// Number of days to skip (1 month = 21 trading days)
    const SKIP_DAYS: usize = 21;

    /// Total lookback period (12 months = 252 trading days)
    const TOTAL_LOOKBACK: usize = 252;
}

impl Factor for LongTermMomentum {
    fn name(&self) -> &str {
        "long_term_momentum"
    }

    fn description(&self) -> &str {
        "12-month (252-day) momentum with 1-month skip - long-term trend persistence"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        Self::TOTAL_LOOKBACK
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

        // Group by symbol and compute momentum with skip period
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                // Price from 1 month ago (t-21)
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit(Self::SKIP_DAYS as i64 + 1)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("price_t_minus_21"),
                // Price from 12 months ago (t-252)
                col("close")
                    .sort_by([col("date")], Default::default())
                    .slice(
                        (lit(0) - lit(Self::TOTAL_LOOKBACK as i64 + 1)).cast(DataType::Int64),
                        lit(1u32),
                    )
                    .first()
                    .alias("price_t_minus_252"),
            ])
            .with_column(
                ((col("price_t_minus_21") / col("price_t_minus_252")) - lit(1.0))
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
    fn test_long_term_momentum_basic() {
        let factor = LongTermMomentum;

        // Create test data with 253 days of prices
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
        // Price goes from 100 to 150 over the full period (50% gain)
        // But we measure from t-252 to t-21
        // At t-252: price = 100.0
        // At t-21: price = 100.0 + 231 * 0.1984 = 145.83
        // Momentum = (145.83 / 100.0) - 1 = 0.4583 (~45.8%)
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

        // Check that momentum is approximately 45.8%
        let momentum = result
            .column("long_term_momentum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (momentum - 0.458).abs() < 0.01,
            "Expected ~0.458, got {}",
            momentum
        );
    }

    #[test]
    fn test_long_term_momentum_metadata() {
        let factor = LongTermMomentum;

        assert_eq!(factor.name(), "long_term_momentum");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }

    #[test]
    fn test_long_term_momentum_skip_period() {
        let factor = LongTermMomentum;

        // Verify that we're using the skip period correctly
        // Create data where the most recent month has a reversal
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

        // Price rises from 100 to 150 from t-252 to t-21
        // Then drops to 140 in the most recent month (t-21 to t)
        let mut prices: Vec<f64> = (0..232).map(|i| 100.0 + i as f64 * 0.2155).collect();
        // At t-21: price should be around 150
        prices.push(150.0);
        // Recent month shows decline
        for i in 233..253 {
            prices.push(150.0 - (i - 232) as f64 * 0.5);
        }

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices.clone(),
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 9, 10).unwrap())
            .unwrap();

        // Momentum should be based on t-21 to t-252, ignoring the recent decline
        let momentum = result
            .column("long_term_momentum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        // Should be around 50% (150/100 - 1), not affected by recent decline
        assert!(
            momentum > 0.4,
            "Momentum should be positive despite recent decline, got {}",
            momentum
        );
    }
}

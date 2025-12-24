//! Time-series momentum factor with volatility scaling.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Time-series momentum factor - direction of past return scaled by inverse volatility.
///
/// Measures the sign of the 252-day return scaled by inverse volatility:
/// `sign(r_{t-252,t}) * (1/σ)`
///
/// where:
/// - `r_{t-252,t}` is the 252-day return
/// - `σ` is the 252-day return volatility
///
/// This factor combines:
/// - Direction: Captures trend direction (up or down)
/// - Risk adjustment: Scales by inverse volatility for risk-adjusted positioning
///
/// Useful for:
/// - Time-series momentum strategies
/// - Risk-adjusted trend following
/// - Volatility-weighted portfolio construction
#[derive(Debug, Clone, Default)]
pub struct TimeSeriesMomentum;

impl Factor for TimeSeriesMomentum {
    fn name(&self) -> &str {
        "time_series_momentum"
    }

    fn description(&self) -> &str {
        "Sign of 252-day return scaled by inverse volatility"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Momentum
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close"]
    }

    fn lookback(&self) -> usize {
        252
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Filter to dates up to and including the target date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())));

        // Compute daily returns using shift
        let with_returns = filtered
            .sort(
                ["symbol", "date"],
                SortMultipleOptions::default().with_order_descending_multi([false, false]),
            )
            .with_column(
                col("close")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("close_lag"),
            )
            .with_column(((col("close") - col("close_lag")) / col("close_lag")).alias("return"))
            .with_column(
                col("close")
                    .shift(lit(self.lookback() as i64))
                    .over([col("symbol")])
                    .alias("close_lagged"),
            );

        // Compute rolling volatility and total return
        let result = with_returns
            .with_column(
                col("return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.lookback(),
                        min_periods: self.lookback(),
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("volatility"),
            )
            .with_column(((col("close") / col("close_lagged")) - lit(1.0)).alias("total_return"))
            .with_column(
                // Compute sign of return
                when(col("total_return").gt(lit(0.0)))
                    .then(lit(1.0))
                    .when(col("total_return").lt(lit(0.0)))
                    .then(lit(-1.0))
                    .otherwise(lit(0.0))
                    .alias("sign_return"),
            )
            .with_column(
                // sign(return) * (1 / max(volatility, 0.0001))
                when(col("volatility").gt(lit(0.0001)))
                    .then(col("sign_return") / col("volatility"))
                    .otherwise(col("sign_return") / lit(0.0001))
                    .alias(self.name()),
            )
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(
                col(self.name())
                    .is_not_null()
                    .and(col(self.name()).is_finite()),
            )
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    #[test]
    fn test_time_series_momentum_basic() {
        let factor = TimeSeriesMomentum;

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
        // Price goes up from 100 to 120 (20% gain)
        let prices: Vec<f64> = (0..253).map(|i| 100.0 + i as f64 * 0.0793).collect();

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 9, 9).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Check that time series momentum is positive (uptrend)
        let ts_mom = result
            .column("time_series_momentum")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            ts_mom > 0.0,
            "Expected positive momentum for uptrend, got {}",
            ts_mom
        );
    }

    #[test]
    fn test_time_series_momentum_metadata() {
        let factor = TimeSeriesMomentum;

        assert_eq!(factor.name(), "time_series_momentum");
        assert_eq!(factor.category(), FactorCategory::Momentum);
        assert_eq!(factor.lookback(), 252);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.required_columns(), &["symbol", "date", "close"]);
    }
}

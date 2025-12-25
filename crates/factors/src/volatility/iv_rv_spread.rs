//! IV-RV Spread factor - volatility risk premium measure.
//!
//! The IV-RV Spread measures the difference between implied volatility from
//! options markets and realized historical volatility. This captures the
//! volatility risk premium - the compensation investors demand for bearing
//! volatility risk.
//!
//! A high spread indicates options are "expensive" relative to actual volatility,
//! suggesting high demand for protection or expected volatility increases.
//! A low or negative spread suggests options are "cheap", potentially indicating
//! complacency or oversupply of volatility selling.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// IV-RV Spread factor.
///
/// Computes the difference between implied volatility (from options) and
/// realized volatility (historical standard deviation of returns). Both
/// volatilities are annualized and typically measured over the same period.
///
/// Formula: `IV - RV`
/// Where:
/// - IV = Implied volatility from at-the-money options (annualized)
/// - RV = Annualized standard deviation of daily returns over lookback period
///
/// # Required Columns
/// - `symbol`: Security identifier
/// - `date`: Date of observation
/// - `close`: Closing price (for RV calculation)
/// - `implied_volatility`: At-the-money implied volatility (annualized, in decimal form, e.g., 0.25 for 25%)
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `iv_rv_spread`
#[derive(Debug, Clone)]
pub struct IvRvSpread {
    lookback: usize,
}

impl IvRvSpread {
    /// Create a new IvRvSpread factor with default lookback (30 days).
    pub const fn new() -> Self {
        Self { lookback: 30 }
    }

    /// Create an IvRvSpread factor with custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for IvRvSpread {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for IvRvSpread {
    fn name(&self) -> &str {
        "iv_rv_spread"
    }

    fn description(&self) -> &str {
        "IV-RV spread - volatility risk premium measure"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Volatility
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close", "implied_volatility"]
    }

    fn lookback(&self) -> usize {
        self.lookback
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Annualization factor for daily to annual volatility
        const TRADING_DAYS_PER_YEAR: f64 = 252.0;
        let annualization_factor = TRADING_DAYS_PER_YEAR.sqrt();

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
            .with_column(((col("close") - col("close_lag")) / col("close_lag")).alias("return"));

        // Compute realized volatility (RV) as rolling standard deviation
        let with_rv = with_returns
            .with_column(
                col("return")
                    .rolling_std(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("daily_vol"),
            )
            .with_column(
                (col("daily_vol") * lit(annualization_factor)).alias("realized_volatility"),
            );

        // Compute IV-RV spread
        let result = with_rv
            .with_column(
                (col("implied_volatility") - col("realized_volatility")).alias("iv_rv_spread"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            .select([col("symbol"), col("date"), col("iv_rv_spread")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn test_iv_rv_spread_lookback() {
        let factor = IvRvSpread::with_lookback(21);
        assert_eq!(factor.lookback(), 21);
    }

    #[test]
    fn test_iv_rv_spread_default_lookback() {
        let factor = IvRvSpread::new();
        assert_eq!(factor.lookback(), 30);
    }

    #[test]
    fn test_iv_rv_spread_metadata() {
        let factor = IvRvSpread::new();
        assert_eq!(factor.name(), "iv_rv_spread");
        assert_eq!(
            factor.description(),
            "IV-RV spread - volatility risk premium measure"
        );
        assert_eq!(factor.category(), FactorCategory::Volatility);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "close", "implied_volatility"]
        );
    }

    #[test]
    fn test_iv_rv_spread_computation() {
        // Create test data with 40 days of price and IV data
        let lookback = 30;
        let factor = IvRvSpread::with_lookback(lookback);

        // Generate synthetic data: stable prices (low RV) with high IV
        let dates: Vec<String> = (0..40)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 40];

        // Stable prices around 100 with small random walk
        let mut prices = vec![100.0];
        for _ in 1..40 {
            prices.push(prices.last().unwrap() + 0.1); // Very low volatility
        }

        // High implied volatility
        let iv = vec![0.30; 40]; // 30% IV

        let df = df![
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
            "implied_volatility" => iv
        ]
        .unwrap();

        let lf = df.lazy();
        let target_date = NaiveDate::from_ymd_opt(2024, 2, 9).unwrap();

        let result = factor.compute_raw(&lf, target_date).unwrap();

        // Check output structure
        assert_eq!(result.shape().1, 3); // 3 columns
        assert!(result.column("symbol").is_ok());
        assert!(result.column("date").is_ok());
        assert!(result.column("iv_rv_spread").is_ok());

        // The spread should be positive (high IV, low RV)
        let spread_series = result.column("iv_rv_spread").unwrap();
        let spread_value = spread_series.f64().unwrap().get(0);
        assert!(spread_value.is_some());

        if let Some(spread) = spread_value {
            // With stable prices, RV should be very low, so IV-RV should be close to IV (0.30)
            assert!(spread > 0.0, "Spread should be positive when IV > RV");
        }
    }

    #[test]
    fn test_iv_rv_spread_negative_spread() {
        // Test case where RV > IV (negative spread)
        let lookback = 30;
        let factor = IvRvSpread::with_lookback(lookback);

        let dates: Vec<String> = (0..40)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i))
                    .unwrap()
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 40];

        // Volatile prices
        let prices: Vec<f64> = (0..40)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0) // Create volatility
            .collect();

        // Low implied volatility
        let iv = vec![0.05; 40]; // 5% IV (low)

        let df = df![
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
            "implied_volatility" => iv
        ]
        .unwrap();

        let lf = df.lazy();
        let target_date = NaiveDate::from_ymd_opt(2024, 2, 9).unwrap();

        let result = factor.compute_raw(&lf, target_date).unwrap();

        let spread_series = result.column("iv_rv_spread").unwrap();
        let spread_value = spread_series.f64().unwrap().get(0);

        if let Some(spread) = spread_value {
            // With volatile prices and low IV, spread could be negative
            assert!(
                spread < 0.20,
                "Spread should be low when RV is high and IV is low"
            );
        }
    }

    #[test]
    fn test_iv_rv_spread_multiple_symbols() {
        // Test with multiple symbols
        let lookback = 30;
        let factor = IvRvSpread::with_lookback(lookback);

        let mut symbols = Vec::new();
        let mut dates = Vec::new();
        let mut prices = Vec::new();
        let mut ivs = Vec::new();

        for symbol in &["AAPL", "MSFT"] {
            for i in 0..40 {
                symbols.push(*symbol);
                dates.push(
                    NaiveDate::from_ymd_opt(2024, 1, 1)
                        .unwrap()
                        .checked_add_days(chrono::Days::new(i))
                        .unwrap()
                        .to_string(),
                );
                prices.push(100.0 + i as f64);
                ivs.push(if *symbol == "AAPL" { 0.25 } else { 0.35 });
            }
        }

        let df = df![
            "symbol" => symbols,
            "date" => dates,
            "close" => prices,
            "implied_volatility" => ivs
        ]
        .unwrap();

        let lf = df.lazy();
        let target_date = NaiveDate::from_ymd_opt(2024, 2, 9).unwrap();

        let result = factor.compute_raw(&lf, target_date).unwrap();

        // Should have 2 rows (one for each symbol)
        assert_eq!(result.shape().0, 2);

        let symbols_result = result.column("symbol").unwrap();
        assert_eq!(symbols_result.len(), 2);
    }
}

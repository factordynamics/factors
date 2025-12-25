//! Kyle's lambda price impact coefficient.
//!
//! Measures the price impact per unit of order flow.
//! Higher lambda indicates less liquid markets with greater price impact.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Kyle's lambda price impact coefficient.
///
/// Estimates the price impact of trading by regressing absolute returns on
/// signed volume (volume weighted by return sign). Lambda represents the
/// price movement per unit of order flow.
///
/// # Interpretation
///
/// - **Higher values**: Greater price impact, less liquid
/// - **Lower values**: Lower price impact, more liquid
///
/// # Computation
///
/// For each security over the lookback period:
/// 1. Calculate returns: `r_t = (close_t - close_{t-1}) / close_{t-1}`
/// 2. Calculate signed volume: `signed_vol_t = volume_t * sign(r_t)`
/// 3. Regress |r_t| on signed_vol_t
/// 4. Lambda = regression coefficient
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `close`: Closing price
/// - `volume`: Daily trading volume
///
/// # References
///
/// - Kyle, A. S. (1985). "Continuous auctions and insider trading," Econometrica 53(6), 1315-1335.
/// - Hasbrouck, J. (2009). "Trading costs and returns for U.S. equities: Estimating effective
///   costs from daily data," Journal of Finance 64(3), 1445-1477.
#[derive(Debug, Clone)]
pub struct KyleLambda {
    lookback: usize,
}

impl KyleLambda {
    /// Creates a new KyleLambda factor with default 20-day lookback.
    pub const fn new() -> Self {
        Self { lookback: 20 }
    }

    /// Creates a KyleLambda factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self { lookback }
    }
}

impl Default for KyleLambda {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for KyleLambda {
    fn name(&self) -> &str {
        "kyle_lambda"
    }

    fn description(&self) -> &str {
        "Price impact coefficient from regressing absolute return on signed volume over 20 days"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "close", "volume"]
    }

    fn lookback(&self) -> usize {
        self.lookback
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let result = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())))
            .sort(
                ["symbol", "date"],
                SortMultipleOptions::default().with_order_descending_multi([false, false]),
            )
            // Calculate lagged close for returns
            .with_column(
                col("close")
                    .shift(lit(1))
                    .over([col("symbol")])
                    .alias("close_lag"),
            )
            // Calculate returns
            .with_column(((col("close") - col("close_lag")) / col("close_lag")).alias("returns"))
            // Calculate absolute returns
            .with_column(col("returns").abs().alias("abs_returns"))
            // Calculate signed volume: volume * sign(returns)
            // sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
            .with_column(
                when(col("returns").gt(lit(0.0)))
                    .then(col("volume"))
                    .when(col("returns").lt(lit(0.0)))
                    .then(-col("volume"))
                    .otherwise(lit(0.0))
                    .alias("signed_volume"),
            )
            // Calculate rolling covariance manually: Cov(X,Y) = E[XY] - E[X]E[Y]
            .with_column((col("abs_returns") * col("signed_volume")).alias("abs_ret_svol_product"))
            .with_column(
                col("abs_ret_svol_product")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("mean_product"),
            )
            .with_column(
                col("abs_returns")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("mean_abs_returns"),
            )
            .with_column(
                col("signed_volume")
                    .rolling_mean(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("mean_signed_volume"),
            )
            // Covariance = E[XY] - E[X]E[Y]
            .with_column(
                (col("mean_product") - (col("mean_abs_returns") * col("mean_signed_volume")))
                    .alias("cov_abs_ret_svol"),
            )
            .with_column(
                col("signed_volume")
                    .rolling_var(RollingOptionsFixedWindow {
                        window_size: self.lookback,
                        min_periods: self.lookback,
                        ..Default::default()
                    })
                    .over([col("symbol")])
                    .alias("var_signed_volume"),
            )
            // Kyle's lambda: Cov / Var
            // Add small epsilon to avoid division by zero
            .with_column(
                (col("cov_abs_ret_svol") / (col("var_signed_volume") + lit(1e-10)))
                    .alias("kyle_lambda_raw"),
            )
            // Handle negative lambda (set to 0 as lambda should be positive)
            .with_column(
                when(col("kyle_lambda_raw").lt(lit(0.0)))
                    .then(lit(0.0))
                    .otherwise(col("kyle_lambda_raw"))
                    .alias("kyle_lambda"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("kyle_lambda")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyle_lambda_metadata() {
        let factor = KyleLambda::new();
        assert_eq!(factor.name(), "kyle_lambda");
        assert_eq!(factor.lookback(), 20);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"close"));
        assert!(factor.required_columns().contains(&"volume"));
    }

    #[test]
    fn test_kyle_lambda_with_custom_lookback() {
        let factor = KyleLambda::with_lookback(15);
        assert_eq!(factor.lookback(), 15);
    }
}

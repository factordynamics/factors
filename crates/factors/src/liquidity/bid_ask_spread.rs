//! Bid-ask spread factor.
//!
//! Measures trading costs through the relative bid-ask spread.
//! Lower spreads indicate more liquid securities.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Bid-ask spread factor.
///
/// Computes the relative bid-ask spread as a percentage of the mid-price.
/// This is the most direct measure of the trading cost incurred by market participants.
///
/// # Interpretation
///
/// - **Higher values**: Less liquid, higher transaction costs
/// - **Lower values**: More liquid, lower transaction costs
///
/// # Computation
///
/// For each security and date:
/// 1. Calculate mid-price: `mid = (ask + bid) / 2`
/// 2. Calculate relative spread: `spread = (ask - bid) / mid`
///
/// # Required Columns
///
/// - `symbol`: Security identifier
/// - `date`: Trading date
/// - `bid`: Bid price
/// - `ask`: Ask price
///
/// # References
///
/// - Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread
///   in an efficient market," Journal of Finance.
///
/// Configuration for BidAskSpread factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidAskSpreadConfig {
    /// Lookback period (typically 1 for current spread).
    pub lookback: usize,
}

impl Default for BidAskSpreadConfig {
    fn default() -> Self {
        Self { lookback: 1 }
    }
}

/// Bid-ask spread factor implementation.
#[derive(Debug, Clone)]
pub struct BidAskSpread {
    config: BidAskSpreadConfig,
}

impl BidAskSpread {
    /// Creates a new BidAskSpread factor with default 1-day lookback.
    pub const fn new() -> Self {
        Self {
            config: BidAskSpreadConfig { lookback: 1 },
        }
    }

    /// Creates a BidAskSpread factor with a custom lookback period.
    pub const fn with_lookback(lookback: usize) -> Self {
        Self {
            config: BidAskSpreadConfig { lookback },
        }
    }
}

impl Default for BidAskSpread {
    fn default() -> Self {
        Self::new()
    }
}

impl Factor for BidAskSpread {
    fn name(&self) -> &str {
        "bid_ask_spread"
    }

    fn description(&self) -> &str {
        "Relative bid-ask spread as percentage of mid-price"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Liquidity
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "bid", "ask"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback
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
            // Calculate mid-price
            .with_column(((col("ask") + col("bid")) / lit(2.0)).alias("mid_price"))
            // Calculate relative spread: (ask - bid) / mid
            // Add small epsilon to avoid division by zero
            .with_column(
                ((col("ask") - col("bid")) / (col("mid_price") + lit(1e-10)))
                    .alias("bid_ask_spread"),
            )
            // Filter to the requested date
            .filter(col("date").eq(lit(date.to_string())))
            // Select output columns
            .select([col("symbol"), col("date"), col("bid_ask_spread")])
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for BidAskSpread {
    type Config = BidAskSpreadConfig;

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

    #[test]
    fn test_bid_ask_spread_metadata() {
        let factor = BidAskSpread::new();
        assert_eq!(factor.name(), "bid_ask_spread");
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(factor.category(), FactorCategory::Liquidity);
        assert!(factor.required_columns().contains(&"bid"));
        assert!(factor.required_columns().contains(&"ask"));
    }

    #[test]
    fn test_bid_ask_spread_with_custom_lookback() {
        let factor = BidAskSpread::with_lookback(5);
        assert_eq!(factor.lookback(), 5);
    }
}

//! Insider Trading factor - Net insider buying activity.
//!
//! This factor measures the balance between insider purchases and sales.
//! Corporate insiders have superior information about their companies'
//! prospects, and their trading activity can signal future performance.
//!
//! # Academic Foundation
//! Seyhun (1986) - "Insiders' Profits, Costs of Trading, and Market Efficiency"
//! Documents that insider purchases, particularly by top executives, predict
//! positive abnormal returns. Insider sales are weaker signals due to various
//! non-informational reasons (diversification, liquidity needs).

use crate::{
    Result,
    registry::FactorCategory,
    traits::{ConfigurableFactor, DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for the Insider Trading factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsiderTradingConfig {
    /// Number of days to look back for insider trading activity.
    /// Default is 63 (approximately 3 months). Use 21 for 1-month, 126 for 6-month windows.
    pub lookback_days: usize,
}

impl Default for InsiderTradingConfig {
    fn default() -> Self {
        Self { lookback_days: 63 }
    }
}

/// Insider Trading factor based on net insider buying activity.
///
/// Computes the net insider buying ratio over a configurable lookback period:
/// `(Insider Buys - Insider Sells) / (Buys + Sells)`
///
/// where:
/// - `Insider Buys` is the total number/value of insider buy transactions
/// - `Insider Sells` is the total number/value of insider sell transactions
///
/// The ratio ranges from -1 (all sells) to +1 (all buys):
/// - Positive values indicate net buying (bullish signal)
/// - Negative values indicate net selling (bearish signal, though weaker)
/// - Values near 0 indicate balanced or no activity
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Transaction filing date
/// - `insider_buys`: Number or value of insider buy transactions
/// - `insider_sells`: Number or value of insider sell transactions
///
/// # Lookback Period
/// Configurable via `lookback_days` (default: 63 trading days)
///
/// # Usage Notes
/// - Insider buys are stronger signals than sells
/// - Focus on purchases by top executives (CEO, CFO, directors)
/// - Cluster of buys from multiple insiders is particularly bullish
/// - Large single transactions are more meaningful than small ones
#[derive(Debug, Clone, Default)]
pub struct InsiderTrading {
    config: InsiderTradingConfig,
}

impl Factor for InsiderTrading {
    fn name(&self) -> &str {
        "insider_net_buying"
    }

    fn description(&self) -> &str {
        "Net insider trading activity - (Buys - Sells) / (Buys + Sells) over 3 months"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Sentiment
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "insider_buys", "insider_sells"]
    }

    fn lookback(&self) -> usize {
        self.config.lookback_days
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Daily
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Calculate the lookback date
        let lookback_date = date
            .checked_sub_days(chrono::Days::new(self.config.lookback_days as u64))
            .unwrap_or(date);

        // Filter data for the lookback window
        let filtered = data
            .clone()
            .filter(col("date").gt(lit(lookback_date.format("%Y-%m-%d").to_string())))
            .filter(col("date").lt_eq(lit(date.format("%Y-%m-%d").to_string())))
            .collect()?;

        // Group by symbol and sum buys/sells over the period
        let result = filtered
            .lazy()
            .group_by([col("symbol")])
            .agg([
                col("date").sort(Default::default()).last().alias("date"),
                col("insider_buys").sum().alias("total_buys"),
                col("insider_sells").sum().alias("total_sells"),
            ])
            .with_column(
                // Total activity
                (col("total_buys") + col("total_sells")).alias("total_activity"),
            )
            .filter(col("total_activity").gt(lit(0.0))) // Filter out no activity before calculation
            .with_column(
                // Calculate net buying ratio: (buys - sells) / (buys + sells)
                ((col("total_buys") - col("total_sells")) / col("total_activity"))
                    .alias(self.name()),
            )
            .select([col("symbol"), col("date"), col(self.name())])
            .filter(col(self.name()).is_not_null())
            .collect()?;

        Ok(result)
    }
}

impl ConfigurableFactor for InsiderTrading {
    type Config = InsiderTradingConfig;

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
    fn test_insider_trading_net_buying() {
        let factor = InsiderTrading::default();

        // Create test data with more buys than sells
        let dates: Vec<String> = (0..10)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i * 3))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 10];
        // 8 buys, 2 sells over the period
        let buys = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let sells = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "insider_buys" => buys,
            "insider_sells" => sells,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);
        assert_eq!(result.width(), 3);

        // Net = (8 - 2) / (8 + 2) = 6 / 10 = 0.6
        let net_buying = result
            .column("insider_net_buying")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (net_buying - 0.6).abs() < 0.01,
            "Expected net buying of 0.6, got {}",
            net_buying
        );
    }

    #[test]
    fn test_insider_trading_net_selling() {
        let factor = InsiderTrading::default();

        // Create test data with more sells than buys
        let dates: Vec<String> = (0..10)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i * 3))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["MSFT"; 10];
        // 2 buys, 8 sells over the period
        let buys = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sells = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "insider_buys" => buys,
            "insider_sells" => sells,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 1);

        // Net = (2 - 8) / (2 + 8) = -6 / 10 = -0.6
        let net_buying = result
            .column("insider_net_buying")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();

        assert!(
            (net_buying + 0.6).abs() < 0.01,
            "Expected net buying of -0.6, got {}",
            net_buying
        );
    }

    #[test]
    fn test_insider_trading_multiple_stocks() {
        let factor = InsiderTrading::default();

        // Create test data for multiple stocks
        let mut dates = Vec::new();
        let mut symbols = Vec::new();
        let mut buys = Vec::new();
        let mut sells = Vec::new();

        // AAPL: heavy buying (80% buys)
        for i in 0..10 {
            dates.push(
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i * 2))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string(),
            );
            symbols.push("AAPL");
            buys.push(if i < 8 { 1.0 } else { 0.0 });
            sells.push(if i >= 8 { 1.0 } else { 0.0 });
        }

        // MSFT: balanced (50/50)
        for i in 0..10 {
            dates.push(
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i * 2))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string(),
            );
            symbols.push("MSFT");
            buys.push(if i % 2 == 0 { 1.0 } else { 0.0 });
            sells.push(if i % 2 == 1 { 1.0 } else { 0.0 });
        }

        // GOOGL: heavy selling (20% buys)
        for i in 0..10 {
            dates.push(
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i * 2))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string(),
            );
            symbols.push("GOOGL");
            buys.push(if i < 2 { 1.0 } else { 0.0 });
            sells.push(if i >= 2 { 1.0 } else { 0.0 });
        }

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "insider_buys" => buys,
            "insider_sells" => sells,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 31).unwrap())
            .unwrap();

        assert_eq!(result.height(), 3);
        assert!(result.column("insider_net_buying").is_ok());

        let syms = result.column("symbol").unwrap().str().unwrap();
        let nets = result.column("insider_net_buying").unwrap().f64().unwrap();

        for i in 0..result.height() {
            let sym = syms.get(i).unwrap();
            let net = nets.get(i).unwrap();

            match sym {
                "AAPL" => assert!(net > 0.5, "AAPL should have strong net buying, got {}", net),
                "MSFT" => assert!(net.abs() < 0.1, "MSFT should be balanced, got {}", net),
                "GOOGL" => assert!(net < -0.5, "GOOGL should have net selling, got {}", net),
                _ => panic!("Unexpected symbol: {}", sym),
            }
        }
    }

    #[test]
    fn test_insider_trading_metadata() {
        let factor = InsiderTrading::default();

        assert_eq!(factor.name(), "insider_net_buying");
        assert_eq!(factor.category(), FactorCategory::Sentiment);
        assert_eq!(factor.lookback(), 63);
        assert_eq!(factor.frequency(), DataFrequency::Daily);
        assert_eq!(
            factor.required_columns(),
            &["symbol", "date", "insider_buys", "insider_sells"]
        );
    }

    #[test]
    fn test_insider_trading_no_activity() {
        let factor = InsiderTrading::default();

        // Create test data with no insider activity
        let dates: Vec<String> = (0..10)
            .map(|i| {
                NaiveDate::from_ymd_opt(2024, 1, 1)
                    .unwrap()
                    .checked_add_days(chrono::Days::new(i * 3))
                    .unwrap()
                    .format("%Y-%m-%d")
                    .to_string()
            })
            .collect();

        let symbols = vec!["AAPL"; 10];
        let buys = vec![0.0; 10];
        let sells = vec![0.0; 10];

        let df = df! {
            "symbol" => symbols,
            "date" => dates,
            "insider_buys" => buys,
            "insider_sells" => sells,
        }
        .unwrap();

        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 31).unwrap())
            .unwrap();

        // Should return empty due to no activity (filtered out)
        assert_eq!(result.height(), 0);
    }
}

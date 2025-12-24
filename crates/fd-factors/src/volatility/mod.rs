//! Volatility factors - measures of risk
//!
//! Volatility factors capture systematic risk exposure (beta) and
//! total risk (historical volatility).

pub mod beta;
pub mod historical_vol;

pub use beta::MarketBeta;
pub use historical_vol::HistoricalVolatility;

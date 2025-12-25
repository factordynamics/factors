//! Volatility factors - measures of risk
//!
//! Volatility factors capture systematic risk exposure (beta),
//! total risk (historical volatility), and volatility risk premium (IV-RV spread).

pub mod beta;
pub mod downside_beta;
pub mod historical_vol;
pub mod idio_vol;
pub mod iv_rv_spread;
pub mod kurtosis;
pub mod max_drawdown;
pub mod parkinson;
pub mod skewness;
pub mod var;

pub use beta::MarketBeta;
pub use downside_beta::DownsideBeta;
pub use historical_vol::HistoricalVolatility;
pub use idio_vol::IdiosyncraticVolatility;
pub use iv_rv_spread::IvRvSpread;
pub use kurtosis::ReturnKurtosis;
pub use max_drawdown::MaxDrawdown;
pub use parkinson::ParkinsonVolatility;
pub use skewness::ReturnSkewness;
pub use var::ValueAtRisk;

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

pub use beta::{MarketBeta, MarketBetaConfig};
pub use downside_beta::{DownsideBeta, DownsideBetaConfig};
pub use historical_vol::{HistoricalVolatility, HistoricalVolatilityConfig};
pub use idio_vol::{IdiosyncraticVolatility, IdiosyncraticVolatilityConfig};
pub use iv_rv_spread::{IvRvSpread, IvRvSpreadConfig};
pub use kurtosis::{ReturnKurtosis, ReturnKurtosisConfig};
pub use max_drawdown::{MaxDrawdown, MaxDrawdownConfig};
pub use parkinson::{ParkinsonVolatility, ParkinsonVolatilityConfig};
pub use skewness::{ReturnSkewness, ReturnSkewnessConfig};
pub use var::{ValueAtRisk, ValueAtRiskConfig};

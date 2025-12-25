#![doc = include_str!("../README.md")]
#![doc(issue_tracker_base_url = "https://github.com/factordynamics/factors/issues/")]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod error;
pub mod growth;
pub mod liquidity;
pub mod momentum;
pub mod quality;
pub mod registry;
pub mod sentiment;
pub mod size;
pub mod standardize;
pub mod traits;
pub mod value;
pub mod volatility;

// Re-export core types
pub use error::{FactorError, Result};
pub use registry::{FactorCategory, FactorInfo, FactorRegistry};
pub use standardize::{cross_sectional_standardize, robust_standardize, winsorize};
pub use traits::{ConfigurableFactor, DataFrequency, Factor, FactorConfig};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

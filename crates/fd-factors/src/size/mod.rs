//! Size factors - measures of company scale
//!
//! Size factors capture the tendency of smaller companies to outperform
//! larger ones (the "size effect").

pub mod enterprise_value;
pub mod log_mcap;
pub mod market_cap;

pub use enterprise_value::EnterpriseValue;
pub use log_mcap::LogMarketCap;
pub use market_cap::MarketCap;

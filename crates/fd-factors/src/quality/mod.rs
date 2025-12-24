//! Quality factors - measures of profitability and financial health
//!
//! Quality factors capture the tendency of high-quality companies with
//! strong profitability and sound finances to outperform.

pub mod leverage;
pub mod margins;
pub mod roa;
pub mod roe;

pub use leverage::Leverage;
pub use margins::ProfitMargin;
pub use roa::Roa;
pub use roe::Roe;

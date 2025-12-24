//! Growth factors - measures of growth rates
//!
//! Growth factors capture companies with accelerating earnings and sales,
//! which may continue to outperform.

pub mod earnings_growth;
pub mod sales_growth;

pub use earnings_growth::EarningsGrowth;
pub use sales_growth::SalesGrowth;

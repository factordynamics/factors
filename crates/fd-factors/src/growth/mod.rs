//! Growth factors - measures of growth rates
//!
//! Growth factors capture companies with accelerating earnings and sales,
//! which may continue to outperform.

pub mod asset_growth;
pub mod book_equity_growth;
pub mod earnings_growth;
pub mod employee_growth;
pub mod sales_growth;

pub use asset_growth::AssetGrowth;
pub use book_equity_growth::BookEquityGrowth;
pub use earnings_growth::EarningsGrowth;
pub use employee_growth::EmployeeGrowth;
pub use sales_growth::SalesGrowth;

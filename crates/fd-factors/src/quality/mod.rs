//! Quality factors - measures of profitability and financial health
//!
//! Quality factors capture the tendency of high-quality companies with
//! strong profitability and sound finances to outperform.

pub mod accruals;
pub mod altman_z;
pub mod asset_turnover;
pub mod cashflow_quality;
pub mod current_ratio;
pub mod earnings_persistence;
pub mod earnings_smoothness;
pub mod gross_profitability;
pub mod interest_coverage;
pub mod leverage;
pub mod margins;
pub mod piotroski;
pub mod quick_ratio;
pub mod roa;
pub mod roe;
pub mod roic;

pub use accruals::AccrualsQuality;
pub use altman_z::AltmanZ;
pub use asset_turnover::AssetTurnover;
pub use cashflow_quality::CashflowQuality;
pub use current_ratio::CurrentRatio;
pub use earnings_persistence::EarningsPersistence;
pub use earnings_smoothness::EarningsSmoothness;
pub use gross_profitability::GrossProfitability;
pub use interest_coverage::InterestCoverage;
pub use leverage::Leverage;
pub use margins::ProfitMargin;
pub use piotroski::Piotroski;
pub use quick_ratio::QuickRatio;
pub use roa::Roa;
pub use roe::Roe;
pub use roic::Roic;

//! Value factors - measures of relative cheapness
//!
//! Value factors capture the tendency of undervalued securities to outperform.
//! Common value metrics include book-to-price, earnings yield, and FCF yield.

pub mod book_to_price;
pub mod earnings_yield;
pub mod fcf_yield;

pub use book_to_price::BookToPrice;
pub use earnings_yield::EarningsYield;
pub use fcf_yield::FcfYield;

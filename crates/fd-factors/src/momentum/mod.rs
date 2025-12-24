//! Momentum factors - measures of trend persistence
//!
//! Momentum factors capture the tendency of securities with strong recent performance
//! to continue outperforming. Different lookback windows capture short, medium, and
//! long-term momentum effects.

pub mod long_term;
pub mod medium_term;
pub mod short_term;

pub use long_term::LongTermMomentum;
pub use medium_term::MediumTermMomentum;
pub use short_term::ShortTermMomentum;

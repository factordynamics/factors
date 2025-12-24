//! Momentum factors - measures of trend persistence
//!
//! Momentum factors capture the tendency of securities with strong recent performance
//! to continue outperforming. Different lookback windows capture short, medium, and
//! long-term momentum effects.

pub mod acceleration;
pub mod high_52week;
pub mod long_term;
pub mod ma_crossover;
pub mod mean_reversion;
pub mod medium_term;
pub mod price_volume_trend;
pub mod residual;
pub mod rsi;
pub mod short_term;
pub mod time_series;
pub mod volatility_breakout;
pub mod volume_momentum;

pub use acceleration::MomentumAcceleration;
pub use high_52week::High52Week;
pub use long_term::LongTermMomentum;
pub use ma_crossover::MACrossover;
pub use mean_reversion::MeanReversion;
pub use medium_term::MediumTermMomentum;
pub use price_volume_trend::PriceVolumeTrend;
pub use residual::ResidualMomentum;
pub use rsi::RSI;
pub use short_term::ShortTermMomentum;
pub use time_series::TimeSeriesMomentum;
pub use volatility_breakout::VolatilityBreakout;
pub use volume_momentum::VolumeMomentum;

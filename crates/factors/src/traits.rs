//! Core trait definitions for factors.
//!
//! All factors implement the [`Factor`] trait, which provides a unified interface
//! for computing factor exposures from market data.

use crate::{FactorCategory, Result};
use chrono::NaiveDate;
use derive_more::Display;
use polars::prelude::*;

/// Data frequency for factor computation.
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataFrequency {
    /// Daily frequency - price-based factors
    Daily,
    /// Quarterly frequency - fundamental-based factors
    Quarterly,
}

/// A factor that can be computed from market data.
///
/// Used by both alpha models (tarifa) and risk models (perth).
/// Factors transform raw market data into standardized exposures
/// that capture systematic sources of return.
pub trait Factor: Send + Sync + std::fmt::Debug {
    /// Unique identifier for this factor.
    ///
    /// Should be snake_case and stable across versions.
    fn name(&self) -> &str;

    /// Human-readable description of what this factor measures.
    fn description(&self) -> &str;

    /// Factor category for grouping and analysis.
    fn category(&self) -> FactorCategory;

    /// Columns required in the input DataFrame.
    ///
    /// The caller must ensure these columns exist before calling `compute`.
    fn required_columns(&self) -> &[&str];

    /// Number of lookback periods needed for computation.
    ///
    /// For daily factors, this is trading days. For quarterly factors,
    /// this is the number of quarters.
    fn lookback(&self) -> usize;

    /// Data frequency required for this factor.
    fn frequency(&self) -> DataFrequency;

    /// Compute raw factor values before standardization.
    ///
    /// Returns a DataFrame with columns: `symbol`, `date`, and the factor name.
    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame>;

    /// Compute standardized factor scores (z-scores).
    ///
    /// This is the primary computation method. It computes raw values
    /// and then applies cross-sectional standardization.
    fn compute(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let raw = self.compute_raw(data, date)?;
        crate::cross_sectional_standardize(&raw, self.name())
    }
}

/// Marker trait for factor configuration types.
///
/// All config types should implement Default, Clone, Send, Sync, and Debug.
pub trait FactorConfig: Default + Clone + Send + Sync + std::fmt::Debug {}

/// A factor that supports runtime configuration.
///
/// This trait extends `Factor` to allow customization of lookback windows,
/// skip periods, and other parameters.
pub trait ConfigurableFactor: Factor {
    /// Configuration type for this factor.
    type Config: FactorConfig;

    /// Create a new factor with the given configuration.
    fn with_config(config: Self::Config) -> Self;

    /// Returns the current configuration.
    fn config(&self) -> &Self::Config;
}

/// Blanket implementation for any type that satisfies the trait bounds.
impl<T: Default + Clone + Send + Sync + std::fmt::Debug> FactorConfig for T {}

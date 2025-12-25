//! Factor registry for discovery and introspection.
//!
//! The registry provides a centralized way to discover, instantiate, and
//! query factors. It supports grouping by category and bulk computation.

use crate::{Factor, Result, traits::DataFrequency};
use chrono::NaiveDate;
use derive_more::Display;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Factor category for grouping related factors.
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FactorCategory {
    /// Momentum - trend persistence factors
    Momentum,
    /// Value - relative valuation factors
    Value,
    /// Quality - profitability and leverage factors
    Quality,
    /// Size - market capitalization factors
    Size,
    /// Volatility - risk and beta factors
    Volatility,
    /// Growth - growth rate factors
    Growth,
    /// Liquidity - trading volume factors
    Liquidity,
    /// Sentiment - analyst and market sentiment factors
    Sentiment,
}

/// Metadata for factor introspection.
#[derive(Debug, Clone)]
pub struct FactorInfo {
    /// Factor name (unique identifier)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Factor category
    pub category: FactorCategory,
    /// Required input columns
    pub required_columns: Vec<String>,
    /// Lookback period
    pub lookback: usize,
    /// Data frequency
    pub frequency: DataFrequency,
}

/// Registry for factor discovery and instantiation.
#[derive(Debug, Default)]
pub struct FactorRegistry {
    factors: HashMap<String, Arc<dyn Factor>>,
}

impl FactorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            factors: HashMap::new(),
        }
    }

    /// Register all standard factors.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Momentum factors
        registry.register(Arc::new(crate::momentum::ShortTermMomentum::default()));
        registry.register(Arc::new(crate::momentum::MediumTermMomentum::default()));
        registry.register(Arc::new(crate::momentum::LongTermMomentum::default()));

        // Value factors
        registry.register(Arc::new(crate::value::BookToPrice::default()));
        registry.register(Arc::new(crate::value::EarningsYield::default()));
        registry.register(Arc::new(crate::value::FcfYield::default()));

        // Quality factors
        registry.register(Arc::new(crate::quality::Roe::default()));
        registry.register(Arc::new(crate::quality::Roa::default()));
        registry.register(Arc::new(crate::quality::ProfitMargin::default()));
        registry.register(Arc::new(crate::quality::Leverage::default()));
        registry.register(Arc::new(crate::quality::GrossProfitability::default()));

        // Size factors
        registry.register(Arc::new(crate::size::LogMarketCap::default()));

        // Volatility factors
        registry.register(Arc::new(crate::volatility::MarketBeta::default()));
        registry.register(Arc::new(crate::volatility::HistoricalVolatility::default()));

        // Growth factors
        registry.register(Arc::new(crate::growth::EarningsGrowth::default()));
        registry.register(Arc::new(crate::growth::SalesGrowth::default()));

        // Liquidity factors
        registry.register(Arc::new(crate::liquidity::TurnoverRatio::default()));
        registry.register(Arc::new(crate::liquidity::AmihudIlliquidity::default()));
        registry.register(Arc::new(crate::liquidity::DollarVolume::default()));
        registry.register(Arc::new(crate::liquidity::BidAskSpread::default()));
        registry.register(Arc::new(crate::liquidity::RollMeasure::default()));
        registry.register(Arc::new(crate::liquidity::CorwinSchultz::default()));
        registry.register(Arc::new(crate::liquidity::ShortInterestRatio::default()));
        registry.register(Arc::new(crate::liquidity::DaysToCover::default()));
        registry.register(Arc::new(crate::liquidity::RelativeVolume::default()));
        registry.register(Arc::new(crate::liquidity::KyleLambda::default()));

        // Sentiment factors
        registry.register(Arc::new(crate::sentiment::AnalystRevisions::default()));

        registry
    }

    /// Register a factor in the registry.
    pub fn register(&mut self, factor: Arc<dyn Factor>) {
        self.factors.insert(factor.name().to_string(), factor);
    }

    /// Get a factor by name.
    pub fn get(&self, name: &str) -> Option<&dyn Factor> {
        self.factors.get(name).map(|f| f.as_ref())
    }

    /// Get factors by category.
    pub fn by_category(&self, category: FactorCategory) -> Vec<&dyn Factor> {
        self.factors
            .values()
            .filter(|f| f.category() == category)
            .map(|f| f.as_ref())
            .collect()
    }

    /// Get all factor metadata.
    pub fn all_info(&self) -> Vec<FactorInfo> {
        self.factors
            .values()
            .map(|f| FactorInfo {
                name: f.name().to_string(),
                description: f.description().to_string(),
                category: f.category(),
                required_columns: f.required_columns().iter().map(|s| s.to_string()).collect(),
                lookback: f.lookback(),
                frequency: f.frequency(),
            })
            .collect()
    }

    /// Get all factor names.
    pub fn names(&self) -> Vec<&str> {
        self.factors.keys().map(|s| s.as_str()).collect()
    }

    /// Compute all factors for a given date.
    ///
    /// Returns a DataFrame with columns: `symbol`, `date`, and one column per factor.
    pub fn compute_all(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let mut result: Option<DataFrame> = None;

        for factor in self.factors.values() {
            let factor_df = factor.compute(data, date)?;

            result = Some(match result {
                Some(df) => df
                    .lazy()
                    .join(
                        factor_df.lazy(),
                        [col("symbol"), col("date")],
                        [col("symbol"), col("date")],
                        JoinArgs::new(JoinType::Inner),
                    )
                    .collect()?,
                None => factor_df,
            });
        }

        result.ok_or_else(|| crate::FactorError::Computation("No factors registered".to_string()))
    }

    /// Number of registered factors.
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
}

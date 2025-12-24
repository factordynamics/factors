//! Piotroski F-Score factor.
//!
//! The Piotroski F-Score is a 9-point composite score that assesses financial strength
//! based on profitability, leverage/liquidity, and operating efficiency signals.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Piotroski F-Score factor.
///
/// The F-Score is a 9-point score (0-9) based on:
///
/// **Profitability (4 points):**
/// - Net Income > 0 (1 point)
/// - Operating Cash Flow > 0 (1 point)
/// - ROA increase from prior period (1 point)
/// - Operating Cash Flow > Net Income (1 point)
///
/// **Leverage/Liquidity (3 points):**
/// - Decrease in long-term debt (1 point)
/// - Increase in current ratio (1 point)
/// - No new equity issuance (1 point)
///
/// **Operating Efficiency (2 points):**
/// - Increase in gross margin (1 point)
/// - Increase in asset turnover (1 point)
///
/// Higher scores (7-9) indicate strong fundamentals, while lower scores (0-2) suggest weakness.
#[derive(Debug, Clone, Default)]
pub struct Piotroski;

impl Factor for Piotroski {
    fn name(&self) -> &str {
        "piotroski_f_score"
    }

    fn description(&self) -> &str {
        "Piotroski F-Score - 9-point composite score assessing financial strength"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Quality
    }

    fn required_columns(&self) -> &[&str] {
        &[
            "symbol",
            "date",
            "net_income",
            "operating_cash_flow",
            "roa",
            "roa_prior",
            "long_term_debt",
            "long_term_debt_prior",
            "current_ratio",
            "current_ratio_prior",
            "shares_outstanding",
            "shares_outstanding_prior",
            "gross_margin",
            "gross_margin_prior",
            "asset_turnover",
            "asset_turnover_prior",
        ]
    }

    fn lookback(&self) -> usize {
        2 // Need current and prior period
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Quarterly
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        let result = data
            .clone()
            .filter(col("date").eq(lit(date.to_string())))
            .select([
                col("symbol"),
                col("date"),
                (
                    // Profitability signals (4 points)
                    col("net_income").gt(lit(0)).cast(DataType::Int32) +
                    col("operating_cash_flow").gt(lit(0)).cast(DataType::Int32) +
                    col("roa").gt(col("roa_prior")).cast(DataType::Int32) +
                    col("operating_cash_flow").gt(col("net_income")).cast(DataType::Int32) +

                    // Leverage/Liquidity signals (3 points)
                    col("long_term_debt").lt(col("long_term_debt_prior")).cast(DataType::Int32) +
                    col("current_ratio").gt(col("current_ratio_prior")).cast(DataType::Int32) +
                    col("shares_outstanding").lt_eq(col("shares_outstanding_prior")).cast(DataType::Int32) +

                    // Operating Efficiency signals (2 points)
                    col("gross_margin").gt(col("gross_margin_prior")).cast(DataType::Int32) +
                    col("asset_turnover").gt(col("asset_turnover_prior")).cast(DataType::Int32)
                ).alias("piotroski_f_score"),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piotroski_metadata() {
        let factor = Piotroski;
        assert_eq!(factor.name(), "piotroski_f_score");
        assert_eq!(factor.lookback(), 2);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.category(), FactorCategory::Quality);
        assert_eq!(factor.required_columns().len(), 16);
    }

    #[test]
    fn test_piotroski_perfect_score() {
        // Create a company with all 9 points
        let df = df![
            "symbol" => ["STRONG"],
            "date" => ["2024-03-31"],
            "net_income" => [10000.0],
            "operating_cash_flow" => [12000.0],
            "roa" => [0.15],
            "roa_prior" => [0.12],
            "long_term_debt" => [50000.0],
            "long_term_debt_prior" => [55000.0],
            "current_ratio" => [2.5],
            "current_ratio_prior" => [2.2],
            "shares_outstanding" => [10000.0],
            "shares_outstanding_prior" => [10000.0],
            "gross_margin" => [0.45],
            "gross_margin_prior" => [0.42],
            "asset_turnover" => [1.3],
            "asset_turnover_prior" => [1.2]
        ]
        .unwrap();

        let factor = Piotroski;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        assert_eq!(result.shape(), (1, 3));

        let scores = result.column("piotroski_f_score").unwrap().i32().unwrap();
        assert_eq!(scores.get(0).unwrap(), 9);
    }

    #[test]
    fn test_piotroski_weak_company() {
        // Create a company with poor fundamentals (low score)
        let df = df![
            "symbol" => ["WEAK"],
            "date" => ["2024-03-31"],
            "net_income" => [-5000.0],  // Negative
            "operating_cash_flow" => [-3000.0],  // Negative
            "roa" => [0.05],
            "roa_prior" => [0.08],  // Declining
            "long_term_debt" => [60000.0],
            "long_term_debt_prior" => [50000.0],  // Increasing
            "current_ratio" => [1.5],
            "current_ratio_prior" => [1.8],  // Declining
            "shares_outstanding" => [12000.0],
            "shares_outstanding_prior" => [10000.0],  // Dilution
            "gross_margin" => [0.25],
            "gross_margin_prior" => [0.30],  // Declining
            "asset_turnover" => [0.9],
            "asset_turnover_prior" => [1.0]  // Declining
        ]
        .unwrap();

        let factor = Piotroski;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let scores = result.column("piotroski_f_score").unwrap().i32().unwrap();

        // Should have a very low score (all signals are negative)
        assert!(scores.get(0).unwrap() <= 2);
    }

    #[test]
    fn test_piotroski_mixed_signals() {
        // Company with some positive and some negative signals
        let df = df![
            "symbol" => ["MIXED"],
            "date" => ["2024-03-31"],
            "net_income" => [8000.0],  // Positive (1 point)
            "operating_cash_flow" => [9000.0],  // Positive (1 point) and > NI (1 point)
            "roa" => [0.10],
            "roa_prior" => [0.12],  // Declining (0 points)
            "long_term_debt" => [45000.0],
            "long_term_debt_prior" => [50000.0],  // Decreasing (1 point)
            "current_ratio" => [2.0],
            "current_ratio_prior" => [2.1],  // Declining (0 points)
            "shares_outstanding" => [10000.0],
            "shares_outstanding_prior" => [10000.0],  // No dilution (1 point)
            "gross_margin" => [0.38],
            "gross_margin_prior" => [0.35],  // Improving (1 point)
            "asset_turnover" => [1.1],
            "asset_turnover_prior" => [1.2]  // Declining (0 points)
        ]
        .unwrap();

        let factor = Piotroski;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())
            .unwrap();

        let scores = result.column("piotroski_f_score").unwrap().i32().unwrap();

        // Should score 6 points based on the positive signals
        assert_eq!(scores.get(0).unwrap(), 6);
    }
}

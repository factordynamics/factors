//! Book-to-price value factor.
//!
//! Measures the ratio of book equity to market capitalization.
//! Higher values indicate potentially undervalued securities.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Book-to-price value factor.
///
/// Computes the ratio of book equity (shareholder equity) to market capitalization.
/// This factor captures the "value premium" - the tendency of securities trading
/// at low multiples of book value to outperform.
///
/// # Formula
///
/// ```text
/// BookToPrice = Book Equity / Market Cap
/// ```
///
/// # Data Requirements
///
/// - `book_equity`: Total shareholder equity (quarterly)
/// - `market_cap`: Market capitalization (daily)
/// - `symbol`: Security identifier
/// - `date`: Date of observation
///
/// # Example
///
/// ```rust,ignore
/// use fd_factors::{Factor, value::BookToPrice};
/// use chrono::NaiveDate;
/// use polars::prelude::*;
///
/// let factor = BookToPrice::default();
/// let data = df![
///     "symbol" => ["AAPL", "MSFT"],
///     "date" => ["2024-03-31", "2024-03-31"],
///     "book_equity" => [50_000_000_000.0, 75_000_000_000.0],
///     "market_cap" => [2_500_000_000_000.0, 2_800_000_000_000.0],
/// ]?.lazy();
///
/// let result = factor.compute(&data, NaiveDate::from_ymd_opt(2024, 3, 31).unwrap())?;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BookToPrice;

impl Factor for BookToPrice {
    fn name(&self) -> &str {
        "book_to_price"
    }

    fn description(&self) -> &str {
        "Book equity divided by market capitalization - measures relative valuation"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Value
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "book_equity", "market_cap"]
    }

    fn lookback(&self) -> usize {
        1
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
                (col("book_equity") / col("market_cap")).alias(self.name()),
            ])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_to_price_basic() {
        let data = df![
            "symbol" => ["AAPL", "MSFT", "GOOGL"],
            "date" => ["2024-03-31", "2024-03-31", "2024-03-31"],
            "book_equity" => [50_000_000_000.0, 75_000_000_000.0, 100_000_000_000.0],
            "market_cap" => [2_500_000_000_000.0, 2_500_000_000_000.0, 2_000_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = BookToPrice;
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 3);

        let values = result.column("book_to_price").unwrap().f64().unwrap();
        assert!((values.get(0).unwrap() - 0.02).abs() < 1e-6);
        assert!((values.get(1).unwrap() - 0.03).abs() < 1e-6);
        assert!((values.get(2).unwrap() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_book_to_price_metadata() {
        let factor = BookToPrice;
        assert_eq!(factor.name(), "book_to_price");
        assert_eq!(factor.category(), FactorCategory::Value);
        assert_eq!(factor.lookback(), 1);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
    }

    #[test]
    fn test_book_to_price_date_filtering() {
        let data = df![
            "symbol" => ["AAPL", "AAPL", "MSFT", "MSFT"],
            "date" => ["2024-03-31", "2024-06-30", "2024-03-31", "2024-06-30"],
            "book_equity" => [50_000_000_000.0, 52_000_000_000.0, 75_000_000_000.0, 78_000_000_000.0],
            "market_cap" => [2_500_000_000_000.0, 2_600_000_000_000.0, 2_500_000_000_000.0, 2_550_000_000_000.0],
        ]
        .unwrap()
        .lazy();

        let factor = BookToPrice;
        let date = NaiveDate::from_ymd_opt(2024, 3, 31).unwrap();
        let result = factor.compute_raw(&data, date).unwrap();

        // Should only get March 31 data
        assert_eq!(result.height(), 2);
        let symbols = result.column("symbol").unwrap().str().unwrap();
        assert_eq!(symbols.get(0).unwrap(), "AAPL");
        assert_eq!(symbols.get(1).unwrap(), "MSFT");
    }
}

//! Employee growth factor implementation.
//!
//! Measures year-over-year employee count growth rate:
//! (Employees_t / Employees_{t-4}) - 1
//!
//! Employee growth can signal business expansion, with rapid hiring
//! potentially indicating growth opportunities or inefficiency.

use crate::{
    Result,
    registry::FactorCategory,
    traits::{DataFrequency, Factor},
};
use chrono::NaiveDate;
use polars::prelude::*;

/// Employee growth factor - year-over-year employee count growth rate.
///
/// Computes (Employees_t / Employees_{t-4}) - 1 using quarterly data.
/// Higher values indicate faster workforce expansion.
///
/// # Required Columns
/// - `symbol`: Stock ticker symbol
/// - `date`: Reporting date (quarterly)
/// - `employees`: Number of employees
///
/// # Returns
/// DataFrame with columns: `symbol`, `date`, `employee_growth`
#[derive(Debug, Clone, Default)]
pub struct EmployeeGrowth;

impl Factor for EmployeeGrowth {
    fn name(&self) -> &str {
        "employee_growth"
    }

    fn description(&self) -> &str {
        "Year-over-year employee count growth rate: (Employees_t / Employees_{t-4}) - 1"
    }

    fn category(&self) -> FactorCategory {
        FactorCategory::Growth
    }

    fn required_columns(&self) -> &[&str] {
        &["symbol", "date", "employees"]
    }

    fn lookback(&self) -> usize {
        4 // 4 quarters for year-over-year comparison
    }

    fn frequency(&self) -> DataFrequency {
        DataFrequency::Quarterly
    }

    fn compute_raw(&self, data: &LazyFrame, date: NaiveDate) -> Result<DataFrame> {
        // Filter data up to the specified date
        let filtered = data
            .clone()
            .filter(col("date").lt_eq(lit(date.to_string())))
            .collect()?;

        // Sort by symbol and date
        let sorted = filtered
            .lazy()
            .sort(
                ["symbol", "date"],
                SortMultipleOptions::default()
                    .with_order_descending_multi([false, false])
                    .with_nulls_last(true),
            )
            .collect()?;

        // Compute year-over-year growth: (Employees_t / Employees_{t-4}) - 1
        let result = sorted
            .lazy()
            .with_column(
                col("employees")
                    .shift(lit(4))
                    .over([col("symbol")])
                    .alias("employees_lag4"),
            )
            .filter(col("date").eq(lit(date.to_string())))
            .with_column(
                ((col("employees") / col("employees_lag4")) - lit(1.0)).alias("employee_growth"),
            )
            .select([col("symbol"), col("date"), col("employee_growth")])
            .collect()?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_employee_growth() {
        // Create test data with quarterly employee counts
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL",
                        "MSFT", "MSFT", "MSFT", "MSFT", "MSFT"],
            "date" => ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01",
                      "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
            "employees" => [10000.0, 10500.0, 11000.0, 11500.0, 12000.0,
                           20000.0, 21000.0, 22000.0, 23000.0, 24000.0]
        ]
        .unwrap();

        let factor = EmployeeGrowth;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // AAPL: (12000 / 10000) - 1 = 0.2 (20% growth)
        // MSFT: (24000 / 20000) - 1 = 0.2 (20% growth)
        assert_eq!(result.height(), 2);
        assert!(result.column("employee_growth").is_ok());

        let growth = result.column("employee_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() - 0.2).abs() < 0.01);
        assert!((growth.get(1).unwrap() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_employee_growth_metadata() {
        let factor = EmployeeGrowth;
        assert_eq!(factor.name(), "employee_growth");
        assert_eq!(factor.category(), FactorCategory::Growth);
        assert_eq!(factor.lookback(), 4);
        assert_eq!(factor.frequency(), DataFrequency::Quarterly);
        assert_eq!(factor.required_columns(), &["symbol", "date", "employees"]);
    }

    #[test]
    fn test_employee_growth_layoffs() {
        // Test with declining employee count (layoffs)
        let df = df![
            "symbol" => ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "date" => ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"],
            "employees" => [10000.0, 9500.0, 9000.0, 8500.0, 8000.0]
        ]
        .unwrap();

        let factor = EmployeeGrowth;
        let result = factor
            .compute_raw(&df.lazy(), NaiveDate::from_ymd_opt(2024, 1, 1).unwrap())
            .unwrap();

        // AAPL: (8000 / 10000) - 1 = -0.2 (-20% decline)
        assert_eq!(result.height(), 1);
        let growth = result.column("employee_growth").unwrap().f64().unwrap();
        assert!((growth.get(0).unwrap() + 0.2).abs() < 0.01);
    }
}

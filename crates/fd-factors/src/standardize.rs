//! Standardization utilities for factor values.
//!
//! Cross-sectional standardization is essential for comparing factor exposures
//! across different securities and time periods.

use crate::Result;
use polars::prelude::*;

/// Cross-sectional z-score standardization.
///
/// Computes z_i = (x_i - mean(x)) / std(x) for each date across all securities.
/// This ensures factor values are comparable across different factors.
///
/// # Arguments
///
/// * `df` - DataFrame with columns: `symbol`, `date`, and `value_column`
/// * `value_column` - Name of the column to standardize
///
/// # Returns
///
/// DataFrame with the value column replaced by its z-scores.
pub fn cross_sectional_standardize(df: &DataFrame, value_column: &str) -> Result<DataFrame> {
    let result = df
        .clone()
        .lazy()
        .with_column(
            (col(value_column) - col(value_column).mean())
                .over([col("date")])
                .alias("centered"),
        )
        .with_column(
            col(value_column)
                .std(1)
                .over([col("date")])
                .alias("std_dev"),
        )
        .with_column((col("centered") / col("std_dev")).alias(value_column))
        .drop(["centered", "std_dev"])
        .collect()?;

    Ok(result)
}

/// Winsorization for outlier handling.
///
/// Clips extreme values to specified percentile bounds. This reduces the
/// impact of outliers on factor standardization.
///
/// # Arguments
///
/// * `df` - DataFrame with the value column
/// * `value_column` - Name of the column to winsorize
/// * `lower_pct` - Lower percentile (e.g., 0.01 for 1st percentile)
/// * `upper_pct` - Upper percentile (e.g., 0.99 for 99th percentile)
///
/// # Returns
///
/// DataFrame with extreme values clipped to percentile bounds.
pub fn winsorize(
    df: &DataFrame,
    value_column: &str,
    lower_pct: f64,
    upper_pct: f64,
) -> Result<DataFrame> {
    let result = df
        .clone()
        .lazy()
        .with_column(
            col(value_column)
                .quantile(lit(lower_pct), QuantileMethod::Linear)
                .over([col("date")])
                .alias("lower_bound"),
        )
        .with_column(
            col(value_column)
                .quantile(lit(upper_pct), QuantileMethod::Linear)
                .over([col("date")])
                .alias("upper_bound"),
        )
        .with_column(
            when(col(value_column).lt(col("lower_bound")))
                .then(col("lower_bound"))
                .when(col(value_column).gt(col("upper_bound")))
                .then(col("upper_bound"))
                .otherwise(col(value_column))
                .alias(value_column),
        )
        .drop(["lower_bound", "upper_bound"])
        .collect()?;

    Ok(result)
}

/// MAD-based robust standardization.
///
/// Computes z_i = (x_i - median(x)) / MAD(x) where MAD is the median absolute
/// deviation. This is more robust to outliers than mean/std standardization.
///
/// # Arguments
///
/// * `df` - DataFrame with columns: `symbol`, `date`, and `value_column`
/// * `value_column` - Name of the column to standardize
///
/// # Returns
///
/// DataFrame with the value column replaced by robust z-scores.
pub fn robust_standardize(df: &DataFrame, value_column: &str) -> Result<DataFrame> {
    // MAD scaling factor for consistency with normal distribution
    const MAD_SCALE: f64 = 1.4826;

    let result = df
        .clone()
        .lazy()
        .with_column(
            col(value_column)
                .median()
                .over([col("date")])
                .alias("median_val"),
        )
        .with_column(
            (col(value_column) - col("median_val"))
                .abs()
                .alias("abs_dev"),
        )
        .with_column(col("abs_dev").median().over([col("date")]).alias("mad_val"))
        .with_column(
            ((col(value_column) - col("median_val")) / (col("mad_val") * lit(MAD_SCALE)))
                .alias(value_column),
        )
        .drop(["median_val", "abs_dev", "mad_val"])
        .collect()?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_sectional_standardize() {
        let df = df![
            "symbol" => ["A", "B", "C", "A", "B", "C"],
            "date" => ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"],
            "value" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ]
        .unwrap();

        let result = cross_sectional_standardize(&df, "value").unwrap();
        let values = result.column("value").unwrap().f64().unwrap();

        // Mean of [1,2,3] = 2, std = 1. So z-scores should be [-1, 0, 1]
        assert!((values.get(0).unwrap() - (-1.0)).abs() < 0.01);
        assert!((values.get(1).unwrap() - 0.0).abs() < 0.01);
        assert!((values.get(2).unwrap() - 1.0).abs() < 0.01);
    }
}

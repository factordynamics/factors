//! CLI for factors unified factor library.
//!
//! This binary provides a command-line interface for discovering, introspecting,
//! and computing factors from the factors library.

use clap::{Parser, Subcommand};
use factors::{FactorCategory, FactorRegistry};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "factors")]
#[command(about = "Unified factor library for quantitative finance", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all available factors
    List,
    /// Show information about a specific factor
    Info {
        /// Factor name
        factor: String,
    },
    /// Compute a factor for a symbol
    Compute {
        /// Stock symbol
        symbol: String,
        /// Factor to compute
        #[arg(long)]
        factor: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let registry = FactorRegistry::with_defaults();

    match cli.command {
        Commands::List => list_factors(&registry),
        Commands::Info { factor } => show_factor_info(&registry, &factor),
        Commands::Compute { symbol, factor } => compute_factor(&registry, &symbol, &factor),
    }
}

/// List all available factors grouped by category.
fn list_factors(registry: &FactorRegistry) {
    let all_info = registry.all_info();

    // Group factors by category
    let mut by_category: HashMap<FactorCategory, Vec<_>> = HashMap::new();
    for info in all_info {
        by_category.entry(info.category).or_default().push(info);
    }

    println!("Available Factors ({} total)\n", registry.len());

    // Sort categories for consistent output
    let mut categories: Vec<_> = by_category.keys().collect();
    categories.sort_by_key(|c| format!("{}", c));

    for category in categories {
        println!("{}:", category);
        let mut factors = by_category.get(category).unwrap().clone();
        factors.sort_by_key(|f| f.name.clone());

        for info in factors {
            println!("  {} - {}", info.name, info.description);
        }
        println!();
    }
}

/// Show detailed information about a specific factor.
fn show_factor_info(registry: &FactorRegistry, factor_name: &str) {
    let all_info = registry.all_info();

    let info = all_info
        .iter()
        .find(|f| f.name == factor_name)
        .unwrap_or_else(|| {
            eprintln!("Error: Factor '{}' not found", factor_name);
            eprintln!("\nAvailable factors:");
            for info in &all_info {
                eprintln!("  {}", info.name);
            }
            std::process::exit(1);
        });

    println!("Factor: {}", info.name);
    println!("Category: {}", info.category);
    println!("Description: {}", info.description);
    println!("Frequency: {:?}", info.frequency);
    println!("Lookback: {} periods", info.lookback);
    println!("Required columns:");
    for col in &info.required_columns {
        println!("  - {}", col);
    }
}

/// Compute a factor for a given symbol.
///
/// This is a placeholder implementation. In a production system, this would:
/// 1. Fetch historical data for the symbol
/// 2. Prepare the data in the required format
/// 3. Call the factor's compute method
/// 4. Display or save the results
fn compute_factor(registry: &FactorRegistry, symbol: &str, factor_name: &str) {
    // Verify the factor exists
    let all_info = registry.all_info();
    let info = all_info
        .iter()
        .find(|f| f.name == factor_name)
        .unwrap_or_else(|| {
            eprintln!("Error: Factor '{}' not found", factor_name);
            eprintln!("\nAvailable factors:");
            for info in &all_info {
                eprintln!("  {}", info.name);
            }
            std::process::exit(1);
        });

    println!("Computing factor '{}' for symbol '{}'", factor_name, symbol);
    println!("\nFactor Details:");
    println!("  Category: {}", info.category);
    println!("  Description: {}", info.description);
    println!("  Lookback: {} periods", info.lookback);
    println!("\nRequired data columns:");
    for col in &info.required_columns {
        println!("  - {}", col);
    }

    println!("\n[PLACEHOLDER] Computation not yet implemented.");
    println!("To compute this factor, you need to:");
    println!(
        "1. Fetch historical {} data for {}",
        frequency_description(&info.frequency),
        symbol
    );
    println!(
        "2. Prepare a DataFrame with required columns: {:?}",
        info.required_columns
    );
    println!(
        "3. Call registry.get(\"{}\").unwrap().compute(&data, date)",
        factor_name
    );
}

/// Get a human-readable description of the data frequency.
const fn frequency_description(freq: &DataFrequency) -> &'static str {
    match freq {
        DataFrequency::Daily => "daily",
        DataFrequency::Quarterly => "quarterly",
    }
}

use factors::DataFrequency;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_not_empty() {
        let registry = FactorRegistry::with_defaults();
        assert!(!registry.is_empty());
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_all_factors_have_info() {
        let registry = FactorRegistry::with_defaults();
        let all_info = registry.all_info();

        assert_eq!(all_info.len(), registry.len());

        for info in all_info {
            assert!(!info.name.is_empty());
            assert!(!info.description.is_empty());
            assert!(!info.required_columns.is_empty());
        }
    }

    #[test]
    fn test_factor_categories() {
        let registry = FactorRegistry::with_defaults();
        let all_info = registry.all_info();

        // Verify we have factors in each category
        let categories: Vec<_> = all_info.iter().map(|f| f.category).collect();

        assert!(categories.contains(&FactorCategory::Momentum));
        assert!(categories.contains(&FactorCategory::Value));
        assert!(categories.contains(&FactorCategory::Quality));
        assert!(categories.contains(&FactorCategory::Size));
        assert!(categories.contains(&FactorCategory::Volatility));
        assert!(categories.contains(&FactorCategory::Growth));
        assert!(categories.contains(&FactorCategory::Liquidity));
    }
}

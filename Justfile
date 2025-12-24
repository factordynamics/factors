set positional-arguments
alias t := test
alias f := fix
alias b := build
alias c := clean

# Default to display help menu
default:
    @just --list

# Runs all CI checks
ci: fix check lychee check-udeps

# Performs lychee checks
lychee:
    @command -v lychee >/dev/null 2>&1 || cargo install lychee
    lychee --config ./lychee.toml .

# Checks formatting, clippy, and tests
check: check-format check-clippy test

# Fixes formatting and clippy issues
fix: format-fix clippy-fix

# Runs tests across workspace
test:
    @command -v cargo-nextest >/dev/null 2>&1 || cargo install cargo-nextest
    RUSTFLAGS="-D warnings" cargo nextest run --workspace --all-features

# Checks formatting
check-format:
    cargo +nightly fmt --all -- --check

# Fixes formatting issues
format-fix:
    cargo fix --allow-dirty --allow-staged
    cargo +nightly fmt --all

# Checks clippy
check-clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Fixes clippy issues
clippy-fix:
    cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged

# Builds the workspace with release
build:
    cargo build --workspace --release

# Cleans the workspace
clean:
    cargo clean

# Checks for unused dependencies
check-udeps:
    @command -v cargo-udeps >/dev/null 2>&1 || cargo install cargo-udeps
    cargo +nightly udeps --workspace --all-features --all-targets

# Runs benchmarks
bench:
    cargo bench --workspace

# Generates documentation
doc:
    cargo doc --workspace --no-deps --open

# Watches for changes and runs tests
watch-test:
    cargo watch -x "nextest run --workspace --all-features"

# Watches for changes and checks
watch-check:
    cargo watch -x "clippy --workspace --all-targets"

# ============================================================================
# Factor Analysis
# ============================================================================

# List all available factors
factors:
    cargo run --package fd-factors-bin -- list

# Compute factor for a symbol
compute symbol factor:
    cargo run --package fd-factors-bin -- compute {{symbol}} --factor {{factor}}

# Show factor information
info factor:
    cargo run --package fd-factors-bin -- info {{factor}}

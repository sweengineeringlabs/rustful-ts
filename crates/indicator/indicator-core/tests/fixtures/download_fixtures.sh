#!/bin/bash
# Download test fixtures from Yahoo Finance
# Usage: ./download_fixtures.sh

FIXTURES_DIR="$(dirname "$0")"

# Function to download symbol data
download_symbol() {
    local symbol=$1
    local start=$2
    local end=$3
    local interval=${4:-1d}
    local output="${FIXTURES_DIR}/${symbol,,}_${interval}.json"

    # Convert dates to timestamps
    start_ts=$(date -d "$start" +%s 2>/dev/null || date -j -f "%Y-%m-%d" "$start" +%s)
    end_ts=$(date -d "$end" +%s 2>/dev/null || date -j -f "%Y-%m-%d" "$end" +%s)

    echo "Downloading $symbol ($interval) from $start to $end..."

    curl -s "https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${start_ts}&period2=${end_ts}&interval=${interval}" \
        -H "User-Agent: Mozilla/5.0" \
        -o "$output"

    if [ -s "$output" ]; then
        echo "  Saved to $output"
    else
        echo "  Failed to download $symbol"
        rm -f "$output"
    fi
}

# Download various symbols for testing
# SPY - S&P 500 ETF (10 years, liquid, good for general testing)
download_symbol "SPY" "2015-01-01" "2025-01-01" "1d"

# BTC-USD - Bitcoin (volatile, good for testing extreme moves)
download_symbol "BTC-USD" "2024-01-01" "2024-12-31" "1d"

# AAPL - Apple (individual stock)
download_symbol "AAPL" "2024-01-01" "2024-12-31" "1d"

# GLD - Gold ETF (different asset class)
download_symbol "GLD" "2024-01-01" "2024-12-31" "1d"

# Intraday data for streaming tests
download_symbol "SPY" "2024-12-01" "2024-12-31" "1h"

echo "Done!"

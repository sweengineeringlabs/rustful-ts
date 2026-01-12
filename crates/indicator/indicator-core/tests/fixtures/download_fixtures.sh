#!/bin/bash
# Download test fixtures from Yahoo Finance
# Usage: ./download_fixtures.sh
#
# Note: Yahoo Finance doesn't have native 4H interval.
# Available intervals: 1m, 2m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo
# For 4H data, download 1H and aggregate in code.
# Intraday data (1H) is limited to ~730 days on Yahoo.

FIXTURES_DIR="$(dirname "$0")"

# Function to download symbol data
# Args: symbol, start_date, end_date, interval, [output_name]
download_symbol() {
    local symbol=$1
    local start=$2
    local end=$3
    local interval=${4:-1d}
    local output_name=${5:-$(echo "${symbol,,}" | tr -d '^' | tr '=' '_')}
    local output="${FIXTURES_DIR}/${output_name}_${interval}.json"

    # Convert dates to timestamps
    start_ts=$(date -d "$start" +%s 2>/dev/null || date -j -f "%Y-%m-%d" "$start" +%s)
    end_ts=$(date -d "$end" +%s 2>/dev/null || date -j -f "%Y-%m-%d" "$end" +%s)

    echo "Downloading $symbol ($interval) from $start to $end..."

    curl -s "https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${start_ts}&period2=${end_ts}&interval=${interval}" \
        -H "User-Agent: Mozilla/5.0" \
        -o "$output"

    if [ -s "$output" ]; then
        # Check if response contains an error
        if grep -q '"error":null' "$output"; then
            local count=$(grep -o '"close":\[' "$output" | wc -l)
            echo "  Saved to $output"
        else
            echo "  Error in response for $symbol"
            cat "$output" | head -c 200
            echo
            rm -f "$output"
        fi
    else
        echo "  Failed to download $symbol"
        rm -f "$output"
    fi
}

echo "=== Downloading Index ETF Fixtures ==="

# SPY - S&P 500 ETF (all available since inception 1993)
download_symbol "SPY" "1993-01-22" "2025-12-31" "1d"
download_symbol "SPY" "2024-01-15" "2025-12-31" "1h"

# QQQ - Nasdaq 100 ETF (proxy for NAS100, since 1999)
download_symbol "QQQ" "1999-03-10" "2025-12-31" "1d" "nas100"
download_symbol "QQQ" "2024-01-15" "2025-12-31" "1h" "nas100"

# GLD - Gold ETF (full history since 2004)
download_symbol "GLD" "2004-11-18" "2025-12-31" "1d"
download_symbol "GLD" "2024-01-15" "2025-12-31" "1h"

echo ""
echo "=== Downloading FAANG Fixtures ==="

# META - Meta/Facebook (IPO May 2012)
download_symbol "META" "2012-05-18" "2025-12-31" "1d"
download_symbol "META" "2024-01-15" "2025-12-31" "1h"

# AAPL - Apple (full history since 1980)
download_symbol "AAPL" "1980-12-12" "2025-12-31" "1d"
download_symbol "AAPL" "2024-01-15" "2025-12-31" "1h"

# AMZN - Amazon (IPO May 1997)
download_symbol "AMZN" "1997-05-15" "2025-12-31" "1d"
download_symbol "AMZN" "2024-01-15" "2025-12-31" "1h"

# NFLX - Netflix (IPO May 2002)
download_symbol "NFLX" "2002-05-23" "2025-12-31" "1d"
download_symbol "NFLX" "2024-01-15" "2025-12-31" "1h"

# GOOGL - Alphabet/Google (IPO Aug 2004)
download_symbol "GOOGL" "2004-08-19" "2025-12-31" "1d"
download_symbol "GOOGL" "2024-01-15" "2025-12-31" "1h"

echo ""
echo "=== Downloading Crypto Fixtures ==="

# BTC-USD - Bitcoin (full history since 2014)
download_symbol "BTC-USD" "2014-09-17" "2025-12-31" "1d"
download_symbol "BTC-USD" "2024-01-15" "2025-12-31" "1h"

echo ""
echo "=== Downloading Forex Fixtures ==="

# Forex pairs - full history available varies by pair
# Note: Yahoo Forex symbols use =X suffix

# EURUSD - Euro/US Dollar (data from ~2003)
download_symbol "EURUSD=X" "2003-01-01" "2025-12-31" "1d" "eurusd"
download_symbol "EURUSD=X" "2024-01-15" "2025-12-31" "1h" "eurusd"

# GBPUSD - British Pound/US Dollar
download_symbol "GBPUSD=X" "2003-01-01" "2025-12-31" "1d" "gbpusd"
download_symbol "GBPUSD=X" "2024-01-15" "2025-12-31" "1h" "gbpusd"

# USDJPY - US Dollar/Japanese Yen
download_symbol "USDJPY=X" "2003-01-01" "2025-12-31" "1d" "usdjpy"
download_symbol "USDJPY=X" "2024-01-15" "2025-12-31" "1h" "usdjpy"

# GBPJPY - British Pound/Japanese Yen
download_symbol "GBPJPY=X" "2003-01-01" "2025-12-31" "1d" "gbpjpy"
download_symbol "GBPJPY=X" "2024-01-15" "2025-12-31" "1h" "gbpjpy"

# EURJPY - Euro/Japanese Yen
download_symbol "EURJPY=X" "2003-01-01" "2025-12-31" "1d" "eurjpy"
download_symbol "EURJPY=X" "2024-01-15" "2025-12-31" "1h" "eurjpy"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Note: 4H timeframe is not available on Yahoo Finance."
echo "To get 4H data, aggregate from 1H bars in your code."
echo ""
ls -lh "$FIXTURES_DIR"/*.json 2>/dev/null | awk '{print $5, $9}'

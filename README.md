# Cached Binance Futures Data Downloader

A lightweight Python module for downloading Binance futures market data with efficient caching support.

## Features

- Download Binance futures data for different timeframes
- Support for custom start and end periods
- Efficient caching mechanism with automatic system document folder integration
- Handles Binance API rate limits automatically
- Supports various timeframes (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
- Minimal dependencies with pure Python implementation

## Installation

```bash
pip install cached-binance-data
```

## Dependencies

- python-binance>=1.0.16: Core Binance API functionality
- requests>=2.26.0: HTTP requests handling
- python-dateutil>=2.8.2: Date manipulation utilities
- pytest>=7.0.0: Testing framework (development only)

## Quick Start

```python
from cached_binance_data import BinanceDataDownloader

# Initialize downloader (uses system document folder by default)
downloader = BinanceDataDownloader()

# Or specify a custom cache directory
# downloader = BinanceDataDownloader(cache_dir="path/to/cache")

# Download XRPUSDT data for a specific period
data = downloader.download(
    symbol="XRPUSDT",
    timeframe="1m",
    start_time="2024-01-01",
    end_time="2024-01-31"
)

# Data is returned as a list of [timestamp, high, low, open, close, volume]
print(f"Downloaded {len(data)} data points")
```

## Cache Structure

By default, data is cached in your system's document folder under `binance_data/`:
- Windows: `C:/Users/<username>/Documents/binance_data/`
- macOS: `/Users/<username>/Documents/binance_data/`
- Linux: `~/Documents/binance_data/` or `~/binance_data/`

Cache files are stored with the following format:
```
SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv
Example: XRPUSDT_1M_20240101_20240131.csv
```

You can specify a custom cache directory when initializing the downloader:
```python
downloader = BinanceDataDownloader(cache_dir="path/to/cache")
```

## Features

- Automatic data chunking for large date ranges
- Smart caching system with system integration
- Handles rate limits gracefully
- Supports all Binance futures trading pairs
- Efficient CSV-based caching for better compatibility

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
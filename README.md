# Binance Futures Data Downloader

A Python module for downloading Binance futures market data with efficient caching support.

## Features

- Download Binance futures data for different timeframes
- Support for custom start and end periods
- Efficient caching mechanism
- Handles Binance API rate limits automatically
- Supports various timeframes (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)

## Installation

```bash
pip install binance-futures-data
```

## Quick Start

```python
from binance_data import BinanceDataDownloader

# Initialize downloader
downloader = BinanceDataDownloader()

# Download XRPUSDT data for a specific period
data = downloader.download(
    symbol="XRPUSDT",
    timeframe="1m",
    start_time="2024-01-01",
    end_time="2024-01-31"
)

# Data is automatically cached in hlocv_cache folder
print(data.head())
```

## Cache Structure

Data is cached in the `hlocv_cache` folder with the following format:
```
SYMBOL_TIMEFRAME_STARTDATE_STARTTIME_ENDDATE_ENDTIME.npy
Example: XRPUSDT_1M_20240101_000000_20240131_235959.npy
```

## Features

- Automatic data chunking for large date ranges
- Smart caching system
- Handles rate limits gracefully
- Supports all Binance futures trading pairs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
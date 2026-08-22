from dotenv import load_dotenv
load_dotenv() # instentiate environment file (making .env variables readable)
import os
import pandas as pd
import databento as db
from enums.utility import INTERVALS

def equity_download(api_key: str, 
                    equity_symbol: str, 
                    timeframe: str,
                    start_date: str, 
                    end_date: str, 
                    data_download_path: str, 
                    file_name: str = ''):
    """
    pulls OHLCV-1m dataset from databento venue databse (download or read)
    """
    client = db.Historical(api_key)
    venue = 'XNAS.ITCH' # Nasdaq Total View venue (collection of databases)
    if timeframe not in INTERVALS:
        raise ValueError(f'timeframe must be one of {INTERVALS}')
    schema = 'OHLCV-' + timeframe 

    # check if file already exists
    file_path = os.path.join(data_download_path, file_name)
    if data_download_path is not None and os.path.exists(file_path):
        return

    try:
        client.timeseries.get_range(
            dataset = venue,
            schema = schema,
            symbols = [equity_symbol],
            start = start_date,
            end = end_date,
            path = file_path
        )
        print(f'successfully downloaded data: {equity_symbol}')
    except Exception as e:
        print(e)
    
if __name__ == "__main__":
    # --- defaults ---
    # uncomment 
    # api_key = '' 
    # data_download_path = '' 

    # comment out when running locally 
    api_key = os.environ.get('DATABENTO_API_KEY') 
    data_download_path = os.environ.get('DATA_DOWNLOAD_PATH')

    symbols = ['T', 'KO'] # BTCUSD uncorrelated, low beta and high daily volume (low spread) stocks
    timeframe = '1m'
    start_date = '2021-07-17'
    end_date = '2025-09-24'

    # --- download --- 
    for s in symbols: 
        equity_download(
            api_key, 
            s, 
            timeframe, 
            data_download_path, 
            start_date,
            end_date
        )
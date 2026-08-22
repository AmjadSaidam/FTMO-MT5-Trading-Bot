from dotenv import load_dotenv
load_dotenv()
import os
import sys
import requests
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enums.utility import INTERVALS
import time

# pull date 
COLUMNS = [
    'date_time', 'open', 'high', 'low', 'close', 'volume', 
    'close_time', 'quote_asset_volume', 'number_trades', 
    'taker_buy_base_vol', 'taker_buy_quote_vol', 'unused'
]

def process_requests(data_path: str,
                     request_input: dict,
                     columns: list,
                     payloads: list):
    """
    """
    resp = requests.get(data_path, params = request_input) # retrive HTTP message, headers is None as no auth required via api secret 
    resp.raise_for_status() # rasie HTTP error if api request unsuccessfull
    batch = resp.json()
    rows = [dict(zip(columns, row)) for row in batch]
    payloads.extend(rows)

    return rows

def get_binance_kline_data(symbol: str,  
                           interval: str, 
                           start_date: pd.Timestamp, 
                           end_date: pd.Timestamp, 
                           request_delay: float = 0.25, 
                           data_path: str = 'https://api.binance.com/api/v3/klines', # endpoint url path
                           data_download_path: str = None, 
                           file_name: str = ''):
    """
    download binance historical kline (OHLCV) data using pagnation from public API endpoint
    """
    # check if file already exists, if so return none
    file_path = os.path.join(data_download_path, file_name)
    if file_path is not None and os.path.exists(file_path):
        return None

    # dates
    freq = pd.Timedelta(value = 5, unit = 'minutes')
    dates = pd.date_range(start = start_date, end = end_date, freq = freq).to_pydatetime().tolist()
    dates = [int(pd.Timestamp(date).timestamp() * 1000) for date in dates] # ms epoch, required by binance api

    if interval not in INTERVALS:
        raise ValueError(f'enter valid interval in {INTERVALS}')

    # default vars
    LIMIT = 1000
    dict_payloads: list[list[dict[str]]] = [] # dict of dict payloads, pre dataframe transformaton

    left = 0
    right = 1000
    
    # inputs for request
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': None,
        'endTime': None,
        'limit': LIMIT,
    }

    while right < len(dates):
        params['startTime'] = dates[left]
        params['endTime'] = dates[right]

        process_requests(data_path, params, COLUMNS, dict_payloads)

        # upadate pointers
        left += 1000
        right += 1000

    else:
        params['startTime'] = dates[left]
        params['endTime'] = dates[-1]

        process_requests(data_path, params, COLUMNS, dict_payloads)

        time.sleep(request_delay) # avoid rate limiting, throttle download

    # filter data
    data = pd.DataFrame(dict_payloads)
    data['date_time'] = pd.to_datetime(data['date_time'], unit = 'ms', utc = True)
    data.set_index('date_time', inplace = True)
    data = data[~data.index.duplicated(keep = 'first')] 

    # save data
    if data_download_path is not None:
        if file_path.endswith('.parquet'):
            data.to_parquet(file_path)
        elif file_path.endswith('.csv'):
            data.to_csv(file_path)

    return None

# download binance data
if __name__ == '__main__':
    # --- pull data default inputs ---
    symbol = 'BTCUSDT'

    # time 
    start_date = '2021-07-17'
    end_date = '2025-09-24'
    
    # data save 
    # save_folder = 'my_training_data_file_path' # uncomment
    save_folder = os.environ.get('WFA_DATA_SAVE_PATH') # comment 
    file_type = '.csv'
    file_name = f'WFA_{symbol}_OHLCV_5m_{start_date.date()}_{end_date.date()}' + file_type

    # load data 
    get_binance_kline_data(symbol, '5m', start_date, end_date, data_download_path = save_folder, file_name = file_name)
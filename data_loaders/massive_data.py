"""
"""
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import time
import requests
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# for OHLCV data
COLUMNS = ['c', 'h', 'l', 'n', 'o', 't', 'v', 'vw']

# pull data
def process_massive_data(url: str,
                         params: dict):
    """
    loads data from massive public api endpoint, following pagination via next_url
    (next_url omits the api key, so it must be re-attached on each follow-up request)
    """
    rows = []
    api_key = params.get('apiKey')

    # massive automatically points dates to range with limit values (unlike binance_data.py with automatic pointing)
    while url:
        resp = requests.get(url, params = params)
        if resp.status_code == 429:
            print(resp.json()) # print payload
            print('1 minute cooldown before next batch download')
            time.sleep(60) # throttle requests, maximum of 5 per minute (free tier)
            continue # re-attempt loop (identical next_url as no new value has been appended) 
        resp.raise_for_status() # print HTTP error (other error)
        batch = resp.json()
        rows.extend(batch.get('results', [])) # subset for results otherwise return empty list if does not exist
        next_url = batch.get('next_url') # break when None 
        url = next_url
        params = {'apiKey': api_key} if next_url else None

    return rows

def get_massive_ohlcv_data(symbol: str,
                           frequency: int,
                           timespan: str,
                           start_date: str,
                           end_date: str,
                           params: dict,
                           data_download_path: str = None,
                           file_name: str = ''):
    """
    download historical forex OHLCV aggregate data from massive public api endpoint
    """
    # check if file already exists
    file_path = os.path.join(data_download_path, file_name)
    if file_path is not None and os.path.exists(file_path):
        return None
    
    if pd.to_datetime(end_date) > pd.to_datetime(start_date) + pd.DateOffset(years = 2):
        start_date = (pd.to_datetime(end_date) - pd.DateOffset(years = 2, days = 1)).strftime('%Y-%m-%d') # account for leap year
        print('WARNING: requesting more base plan data histroy of 2 years')

    # download json data
    url = f'https://api.massive.com/v2/aggs/ticker/{symbol}/range/{frequency}/{timespan}/{start_date}/{end_date}'
    rows = process_massive_data(url, params)

    # transform
    data = pd.DataFrame(rows)[COLUMNS] # fix field orderings
    data = data.drop(labels = ['n', 'vw'], axis = 1)
    data = data.rename(columns = {'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 't': 'datetime', 'v': 'volume'})

    data['datetime'] = pd.to_datetime(data['datetime'], unit = 'ms', utc = True)
    data.set_index('datetime', inplace = True)
    data = data[~data.index.duplicated(keep = 'first')]

    # save locally
    if data_download_path is not None:
        if file_path.endswith('.parquet'):
            data.to_parquet(file_path)
        elif file_path.endswith('.csv'):
            data.to_csv(file_path)

    return None

if __name__ == '__main__':
    # --- pull forex data ---
    # default params
    symbol = 'C:' + 'GBPUSD' # currency prefix
    frequency = 1
    timespan = 'minute'
    start_date = '2021-07-17'
    end_date = '2025-09-24'
    api_key = os.environ.get('MASSIVE_API_KEY')
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000, # max bars per request
        'apiKey': api_key
    }

    # data save
    save_folder = os.environ.get('DATA_DOWNLOAD_PATH')
    file_type = '.csv'
    file_name = f'{symbol.replace(":", "")}_OHLCV_1m_{start_date}_{end_date}' + file_type # remove invalid path charachter ':'

    # load data
    get_massive_ohlcv_data(symbol, frequency, timespan, start_date, end_date, params, save_folder, file_name)
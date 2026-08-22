import kagglehub as kg

# run from terminal: python kaggle_data.py
if __name__ == '__main__':
    try:
        data_path = '/Users/amjadsaidam/Desktop/Quant stuff /FTMO/Trading Strategy One /data'
        path = kg.dataset_download(handle = 'shivaverse/btcusdt-5-minute-ohlc-volume-data-2017-2025', output_dir = data_path)    
        print('successfully saved data to:\n', data_path)
    except Exception as e:
        print('falied download:\n', e)    
    
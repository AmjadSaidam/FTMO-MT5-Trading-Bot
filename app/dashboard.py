"""
streamlit dashboard 

to run call: streamlit run app/dashboard.py
"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_folders = ['code', 'data_loaders'] # code/*.py modules cross-import each other by bare module name (eg. "from risk_management import atr"), so their own folder must be on sys.path too
target_paths = [os.path.join(ROOT, dest) for dest in target_folders]
for p in ([ROOT] + target_paths):
    if p not in sys.path:
        sys.path.insert(0, p)
# general
from collections import defaultdict
import streamlit as st 
import pandas as pd
import numpy as np
import databento as db
# .py
from data_loaders import databento_data, binance_data, massive_data
import code.backtest_engine as backtest 
import code.trading_strategies as ts
import code.challenge_validation as cv
import code.optimisation as opt
from code.regime_filter import r2, categorise_regime
# from code.performance_metrics import sharpe_ratio
from code.risk_management import RandomForestVol
# from code.analytics import prob_consec_win, prob_until_breakdown
# enums 
from enums.challenge_tradeable_assets import TRADEABLE_FOREX, TRADEABLE_CRYPTOCURRENCY, TRADEABLE_EQUITY
# eda 
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff
import plotly.express as px

# =============================================
# Helper Functions
# =============================================
def multi_session_state(states_types: dict[str]):
    for state, type in states_types.items():
        if state not in st.session_state:
            setattr(st.session_state, state, type) 
        else:
            pass

# =============================================
# Title
# =============================================
st.set_page_config(layout = 'wide')
st.title('TRADING STRATEGY SIGNAL VERIFICATION DASHBOARD')

# =============================================
# API 
# =============================================
databento_api_secret = os.environ.get('DATABENTO_API_KEY')
databento_api_key = st.sidebar.text_input('Databento API-key', databento_api_secret)

massive_api_secret = os.environ.get('MASSIVE_API_KEY')
massive_api_key = st.sidebar.text_input('Massive API-key', massive_api_secret)

# =============================================
# Backtest Settings
# =============================================
wfa_trainset_length_days = st.sidebar.number_input('WFA Training Set Length (days)', 20) # identical value backtested, see notebooks/backtest.ipynb
wfa_training_end_date = st.sidebar.date_input('WFA Training Set End Date', pd.to_datetime('2026-08-16') - pd.Timedelta(days = 1)) # trail by 1 day, live feed lags real-time and a partial current day shouldn't be in the training set
wfa_training_start_date = wfa_training_end_date - pd.Timedelta(value = wfa_trainset_length_days, unit = 'D')

# =============================================
# Data 
# =============================================
data_download_path = os.environ.get('WFA_DATA_SAVE_PATH')

asset_forex = st.sidebar.text_input('FOREX', value = 'GBP:USD')
asset_equity = st.sidebar.text_input('EQUITY', value = 'KO')
asset_crypto = st.sidebar.text_input('CRYPTOCURRENCY', value = 'BTCUSD')

# check valid tickers 
if any(ticker not in (TRADEABLE_FOREX + TRADEABLE_EQUITY + TRADEABLE_CRYPTOCURRENCY) for ticker in set([asset_forex, asset_equity, asset_crypto])):
    st.error(f'Non available symbol, must be one of {TRADEABLE_FOREX} or {TRADEABLE_EQUITY} or {TRADEABLE_CRYPTOCURRENCY}')

# FOREX DATA ----------------------------------
massive_file_type = '.csv'
forex_data_name = f'WFA_{asset_forex.replace(":", "")}_OHLCV_1m_{wfa_training_start_date}_{wfa_training_end_date}' + massive_file_type
forex_data_path = os.path.join(data_download_path, forex_data_name)
massive_params = {
    'adjusted': 'true',
    'sort': 'asc',
    'limit': 50000, # max bars per request
    'apiKey': massive_api_key
}
massive_data.get_massive_ohlcv_data('C:' + asset_forex.replace(':', ''), 1, 'minute', wfa_training_start_date, wfa_training_end_date, params = massive_params, data_download_path = data_download_path, file_name = forex_data_name)
forex_data = pd.read_csv(forex_data_path, index_col = 'datetime', parse_dates = ['datetime'])
forex_data = ts.lower_to_higher_timeframe(forex_data, '1min', '5min')

# MULTITIMEFRAME DATA
# get lower time frame data
h1_data = ts.lower_to_higher_timeframe(forex_data, '5min', '1h')
h4_data = ts.lower_to_higher_timeframe(forex_data, '5min', '4h')
h8_data = ts.lower_to_higher_timeframe(forex_data, '5min', '8h')

# allign and merge higher resolution data to lower time frame 
multi_timeframe_data = ts.concatenate_timeframe_data(forex_data, [h1_data, h4_data, h8_data]).dropna()

# add NY and LDN sesssion state columns
multi_timeframe_data = ts.add_session_flags(multi_timeframe_data) # copy to avoid editing original dataframe

# EQUITY DATA ----------------------------------
databento_file_type = '.csv'
equity_file_name = f'WFA_{asset_equity}_OHLCV_1m_{wfa_training_start_date}_{wfa_training_end_date}' + databento_file_type
equity_data_path = os.path.join(data_download_path, equity_file_name)
databento_data.equity_download(databento_api_key, asset_equity, '1m', wfa_training_start_date, wfa_training_end_date, data_download_path, equity_file_name)
equity_data = db.DBNStore.from_file(equity_data_path).to_df().drop(labels = ['symbol', 'rtype', 'publisher_id', 'instrument_id'], axis = 1)

# CRYPTO DATA ----------------------------------
# first download data using binance_data.py 
binance_file_type = '.csv'

crypto_file_name = f'WFA_{asset_crypto}_OHLCV_5m_{wfa_training_start_date}_{wfa_training_end_date}' + binance_file_type
crypto_data_path = os.path.join(data_download_path, crypto_file_name)
# download (binance spot symbols are quoted in USDT, not bare USD)
binance_data.get_binance_kline_data(asset_crypto.replace('USD', 'USDT'), '5m', wfa_training_start_date, wfa_training_end_date, data_download_path = data_download_path, file_name = crypto_file_name) # download
crypto_data = pd.read_csv(crypto_data_path, index_col = 'date_time', parse_dates = ['date_time']) # read file

# DATA JOIN ----------------------------------
# equity data join to forex base index
equity_data = ts.standerdise_join_data(forex_data, equity_data, '1min')

# crypto data join to forex base index 
crypto_data = ts.standerdise_join_data(forex_data, crypto_data)

# strategy data ----------------------------------
st.subheader('ALLIGNED DATA')

with st.expander('FOREX STRATEGY'):
    st.dataframe(multi_timeframe_data)

with st.expander('EQUITY STRATEGY'):
    st.dataframe(equity_data)

with st.expander('CRYPOTCURRENCY STRATEGY'):
    st.dataframe(crypto_data)

# =============================================
# Walk Forward Analysis (WFA)
# =============================================
st.header('LAST STEP WALK FORWARD ANALYSIS')

# strategy backtest ----------------------------------
data = [multi_timeframe_data, equity_data, crypto_data]
data_keys = ['forex', 'equity', 'crypto']
forex = [True, False, False]
strats = [ts.session_breakout, ts.vwap_breakout, ts.bollinger_band_cdv]
strats_kwargs = [{}, {}, {}]
commissions = [0.0, 0.00004, 0.00065] 
assets_trading_full_year = [True, False, False]

account_balance = 1e4
account_balance_per_strat = account_balance / len(data)

opt_atr = 2.0
opt_rr = 2.0

atr_multipliers = np.linspace(1.0, 3.0, 5)
risk_rewards = np.linspace(1.0, 3.0, 5)

# default vars
backtest_state_types = {
    'run_backtest': False, 
    'trained_ensembles': {},
    'opt_vars': defaultdict(dict)
}
multi_session_state(backtest_state_types)

# optimisation ----------------------------------
if st.button('Run WFA on Most Recent Training Test Split'):
    if st.session_state.run_backtest is False:
        # loop over all asset/strategy pairs
        for i in range(len(data)):
            gd_inputs = opt.grid_search_params(data = data[i],
                                               strat = strats[i],
                                               strat_kwargs = strats_kwargs[i],
                                               account_balance = account_balance_per_strat,
                                               atr_multilpiers = atr_multipliers,
                                               risk_rewards = risk_rewards)
            gd_output = opt.run_grid_search(gd_inputs)
            gd_summary_stats = opt.grid_search_statistics(gd_output,
                                                          risk_rewards,
                                                          atr_multipliers)
            # optimal parameters
            opt_df  = gd_summary_stats['opt_df']

            # ensemble model (features identical to backtest_engine.backtest_strategy's point_features schema)
            ml_features = opt_df.loc[:, ['atr_pct', 'regime', 'breakout_conv', 'rel_volume']]
            ensemble = RandomForestVol(ml_features, opt_df['ml_label'], 'regime')
            model_tr = ensemble.train_model(train_split = 1) # train to get .predict method

            # logs
            model_key = data_keys[i]
            st.session_state.trained_ensembles[model_key] = ensemble.model
            st.session_state.opt_vars[model_key]['atr'] = gd_summary_stats['opt_atr']
            st.session_state.opt_vars[model_key]['rr'] = gd_summary_stats['opt_rr']

trained_ensembles = st.session_state.trained_ensembles
opt_vars = st.session_state.opt_vars

# print optimal variables
st.dataframe(pd.DataFrame({
    'ATR_MULT*': [opt_vars['forex']['atr'], opt_vars['equity']['atr'], opt_vars['crypto']['atr']], 
    'RR*': [opt_vars['forex']['rr'], opt_vars['equity']['rr'], opt_vars['crypto']['rr']]
}, index = [asset_forex, asset_equity, asset_crypto]))

# =============================================
# Strategy Signal Verification
# =============================================
st.header('SIGNAL VERIFICATION')

# FOREX ----------------------------------
st.subheader('FOREX ENSEMBLE')

forex_slope_coeff = st.number_input('Regime (30 day SLR R2)', value = 0.0, min_value = 0.0, max_value = 1.0, key = 'forex_regime_input')
forex_atr_pct = st.number_input('ATR % (ATR / Close)', value = 0.0, key = 'forex_atr_pct_input')
forex_breakout_conv = st.number_input('Breakout Conviction ((close-open)/(high-low))', value = 0.0, key = 'forex_breakout_conv_input')
forex_rel_volume = st.number_input('Relative Volume (30 bar rolling mean volume)', value = 0.0, key = 'forex_rel_volume_input')
forex_regime = categorise_regime(np.array([forex_slope_coeff]))[0]

forex_valid_trade = {
    'forex_valid_trade': None
}
multi_session_state(forex_valid_trade)
if st.button('Validate Trade', key = 'forex_validate_button'):
    if st.session_state.forex_valid_trade is None:
        st.session_state.forex_valid_trade = trained_ensembles['forex'].predict(
            pd.DataFrame({'atr_pct': [forex_atr_pct], 'regime': [forex_regime], 'breakout_conv': [forex_breakout_conv], 'rel_volume': [forex_rel_volume]})
        )
if bool(st.session_state.forex_valid_trade):
    st.success(f'TRADE VALID')
else:
    st.error(f'TRADE INVALID')

# EQUITY ----------------------------------
st.subheader('EQUITY ENSEMBLE')

equity_slope_coeff = st.number_input('Regime (30 day SLR R2)', value = 0.0, min_value = 0.0, max_value = 1.0, key = 'equity_regime_input')
equity_atr_pct = st.number_input('ATR % (ATR / Close)', value = 0.0, key = 'equity_atr_pct_input')
equity_breakout_conv = st.number_input('Breakout Conviction ((close-open)/(high-low))', value = 0.0, key = 'equity_breakout_conv_input')
equity_rel_volume = st.number_input('Relative Volume (30 bar rolling mean volume)', value = 0.0, key = 'equity_rel_volume_input')
equity_regime = categorise_regime(np.array([equity_slope_coeff]))[0]

equity_valid_trade = {
    'equity_valid_trade': None
}
multi_session_state(equity_valid_trade)
if st.button('Validate Trade', key = 'equity_validate_button'):
    if st.session_state.equity_valid_trade is None:
        st.session_state.equity_valid_trade = trained_ensembles['equity'].predict(
            pd.DataFrame({'atr_pct': [equity_atr_pct], 'regime': [equity_regime], 'breakout_conv': [equity_breakout_conv], 'rel_volume': [equity_rel_volume]})
        )
if bool(st.session_state.equity_valid_trade):
    st.success(f'TRADE VALID')
else:
    st.error(f'TRADE INVALID')

# CRYPTO ----------------------------------
st.subheader('CRYPTO ENSEMBLE')

crypto_slope_coeff = st.number_input('Regime (30 day SLR R2)', value = 0.0, min_value = 0.0, max_value = 1.0, key = 'crypto_regime_input')
crypto_atr_pct = st.number_input('ATR % (ATR / Close)', value = 0.0, key = 'crypto_atr_pct_input')
crypto_breakout_conv = st.number_input('Breakout Conviction ((close-open)/(high-low))', value = 0.0, key = 'crypto_breakout_conv_input')
crypto_rel_volume = st.number_input('Relative Volume (30 bar rolling mean volume)', value = 0.0, key = 'crypto_rel_volume_input')
crypto_regime = categorise_regime(np.array([crypto_slope_coeff]))[0]

crypto_valid_trade = {
    'crypto_valid_trade': None
}
multi_session_state(crypto_valid_trade)
if st.button('Validate Trade', key = 'crypto_validate_button'):
    if st.session_state.crypto_valid_trade is None:
        st.session_state.crypto_valid_trade = trained_ensembles['crypto'].predict(
            pd.DataFrame({'atr_pct': [crypto_atr_pct], 'regime': [crypto_regime], 'breakout_conv': [crypto_breakout_conv], 'rel_volume': [crypto_rel_volume]})
        )
if bool(st.session_state.crypto_valid_trade):
    st.success(f'TRADE VALID')
else:
    st.error(f'TRADE INVALID')
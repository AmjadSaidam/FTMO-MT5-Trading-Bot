"""
Thin Wrapper of MetaTrader5 functions 

no position-sizing/risk-management or stratey logic

run 'uv pip install MetaTrader5' on Window python < 3.12, no macos or linux build (script must be ran on windows machine)

errors raised otherwise canot be tested without executing live mt5 requests
"""
import os
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd 
# .py
import mt5_errors

def connect_account():
    """establish connection to mt5 terminal, and login into account"""
    mt5.shutdown() # close any stale prior session before reconnecting, no-op if none exists
    env = lambda x: os.environ.get(x)
    conn = mt5.initialize(path = env('META_PATH'),
                          login = env('META_LOGIN'),
                          password = env('META_PASSWORD'),
                          server = env('META_SERVER'))
    if not conn:
        err_tup = mt5_errors.MT5error()
        raise mt5_errors.MT5ConnectionError(f'connection failed with error code {err_tup[0]}: {err_tup[1]}')

def get_latest_bar_time(symbol: str, timeframe: str = mt5.TIMEFRAME_M5) -> int:
    """open time (epoch seconds) of the most recent bar, cheap 1-bar fetch used to detect a new bar close without pulling the full history window"""
    rates = mt5.copy_rates_from(symbol, timeframe, datetime.today(), 1)
    if rates is None:
        err_tup = mt5_errors.MT5error()
        raise mt5_errors.MT5RatesError(f'failed to request latest bar with error code {err_tup[0]}: {err_tup[1]}')

    return rates[-1]['time']

def get_latest_bars_dates(symbol: str,
                          timeframe: str = mt5.TIMEFRAME_M5,
                          date_from: pd.Timestamp | None = None,
                          count: int = int(60 / 5 * 8)):
    """gets symbol metadata using dates, default pulls 8hours of 5 minute data"""
    if date_from is None:
        date_from = datetime.today()

    rates = mt5.copy_rates_from(symbol,
                                timeframe,
                                date_from,
                                count)
    if rates is None:
        err_tup = mt5_errors.MT5error()
        raise mt5_errors.MT5RatesError(f'failed to request data with error code {err_tup[0]}: {err_tup[1]}')

    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit = 's')

    return rates_frame

def send_order(trade_magic_id: int,
               symbol: str, 
               volume: float,
               stop_loss: float, 
               take_profit: float,
               type: str = 'LONG', 
               action = mt5.TRADE_ACTION_DEAL, 
               type_filling = mt5.ORDER_FILLING_FOK,
               **kwargs): 
    """standard order to mt5 server/broker"""
    if type in ['LONG', 'SHORT']:
        if type == 'LONG':
            type = mt5.ORDER_TYPE_BUY
        if type == 'SHORT':
            type = mt5.ORDER_TYPE_SELL
    else:
        raise mt5_errors.MT5OrderError(f"Order 'type' default field not of valid types 'LONG' or 'SHORT'")

    request = {
        "action": action,
        "magic": trade_magic_id,
        "symbol": symbol,
        "volume": volume,
        "type": type,
        "sl": stop_loss,
        "tp": take_profit,
        "type_filling": type_filling,
        **kwargs,
    }
    order = mt5.order_send(request)

    # call None first, is order None has no attributes (retcode)
    if order is None:
        err_tup = mt5_errors.MT5error()
        raise mt5_errors.MT5OrderError(f'Order send failed with error code {err_tup[0]}: {err_tup[1]}')
    
    if order.retcode != mt5.TRADE_RETCODE_DONE:
        raise mt5_errors.MT5OrderError(f'order rejected (retcode = {order.retcode}: {order.comment})')

    return order

def get_positions(ticket: int | None = None,
                  symbol: str | None = None, 
                  magic: int | None = None):
    """returns position named tuple, one to many mapping between position and deals; each trade has one position_id and at least one ecah ticket id"""
    kwargs = {}
    if ticket is not None:
        kwargs['ticket'] = ticket
    if symbol is not None:
        kwargs['symbol'] = symbol
    
    positions = mt5.positions_get(**kwargs) # validates is passed 

    if positions is None: 
        err_tup = mt5_errors.MT5error()
        raise mt5_errors.MT5PositionError(f'failed to retrieve position with error code {err_tup[0]}: {err_tup[1]}')

    if magic is not None:
        positions = tuple(p for p in positions if (p.magic == magic)) # trade attributes

    return positions 

def close_position(position):
    """full volume strategy position close at market current price"""
    tick = mt5.symbol_info_tick(position.symbol)
    close_type = mt5.ORDER_TYPE_SELL if (position.type == mt5.ORDER_TYPE_BUY) else mt5.ORDER_TYPE_BUY
    price = tick.bid if (close_type == mt5.ORDER_TYPE_SELL) else tick.ask

    request = {
        'action': mt5.TRADE_ACTION_DEAL, 
        'position': position.ticket, 
        'symbol': position.symbol,
        'volume': position.volume, 
        'type': close_type, 
        'price': price,
        'deviation': 20, 
        'magic': position.magic, 
        'type_filling': mt5.ORDER_FILLING_FOK,
    }
    order = mt5.order_send(request)

    if order is None:
        err_tup = mt5_errors.MT5error()
        raise mt5_errors.MT5OrderError(f'Position close failed with error code {err_tup[0]}: {err_tup[1]}')
    
    if order.retcode != mt5.TRADE_RETCODE_DONE:
        raise mt5_errors.MT5OrderError(f'Position close rejected (retcode = {order.retcode}): {order.comment}')
    
    return order

def get_deal_profit(ticket):
    """net realised profit (price P&L + commission + swap) summed across all deals for a position"""
    deals = mt5.history_deals_get(position = ticket)

    if deals is None:
        err_tup = mt5_errors.MT5error()
        raise mt5_errors.MT5PositionError(f'failed to retrieve deal history with error code {err_tup[0]}: {err_tup[1]}')

    return sum(deal.profit + deal.commission + deal.swap for deal in deals)
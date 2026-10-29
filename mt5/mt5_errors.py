"""
Constructors for MT5 connection errors 

constructors print raised error when called in except 
"""
import MetaTrader5 as mt5

def MT5error():
    """"""
    return mt5.last_error()

class MT5ConnectionError(Exception):
    pass

class MT5RatesError(Exception):
    pass

class MT5PositionError(Exception):
    pass

class MT5OrderError(Exception):
    pass
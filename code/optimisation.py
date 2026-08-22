import pandas as pd 
import numpy as np
import backtest_engine as backtest
import challenge_validation as cv

# get list of dictionaries with parameters
def grid_search_params(data: pd.DataFrame,
                       strat: backtest.SignalFunc, 
                       strat_kwargs: dict, 
                       account_balance: float, 
                       atr_multilpiers: list[float], 
                       risk_rewards: list[float]):
    """
    """
    inputs = []
    for atr_m in atr_multilpiers:
        for rr in risk_rewards:
            inputs.append(
                {
                    'data': data, 
                    'account_balance': account_balance,
                    'signal_fc': strat,
                    'signal_kwargs': strat_kwargs, 
                    'atr_multiplier': atr_m, 
                    'risk_reward': rr, 
                    'consecutive_loss_before_skip': 3, 
                }
            )
    
    return inputs

# map function to input iterbale
def run_grid_search(training_params: dict[str]):
    """
    """
    return list(map(lambda input: backtest.backtest_strategy(**input)['data'], training_params))

# get final returns 
def grid_search_statistics(grid_search_output: list[pd.DataFrame], 
                           grid_search_x: list[float], 
                           grid_search_y: list[float]) -> dict[pd.DataFrame]:
    """
    """
    n, m = len(grid_search_x), len(grid_search_y)
    final_returns = np.zeros((n, m))

    max_challenge_score = -np.inf
    opt_atr = None 
    opt_rr = None 
    opt_df = None
    
    for i in range(n):
        for j in range(m):
            df = grid_search_output[i*m + j]
            # instentiate challenge function 
            chall_val = cv.PropFirmValidation(df['adjusted_returns'], df['equity'])
            chall_val.strategy_challenge_summary()

            # update to find best data 
            if (chall_val.strategy_score is not None) and chall_val.strategy_score > max_challenge_score: 
                max_challenge_score = chall_val.strategy_score

                opt_atr = grid_search_x[i]
                opt_rr = grid_search_y[j]
                opt_df =  df 
            
            # logs
            equity = (df.loc[:, 'adjusted_returns'] + 1).cumprod() - 1
            ror = equity.iloc[-1]
            final_returns[i, j] = ror # get final cumulative return

    # final dataframe 
    x_label = [str(i) for i in grid_search_x] # atr multipliers 
    y_label = [str(j) for j in grid_search_y] # risk rewards
    gd_ror = pd.DataFrame(final_returns, columns = y_label, index = x_label)
    
    return {
        'ror': gd_ror, 
        'opt_atr': opt_atr,
        'opt_rr': opt_rr, 
        'opt_df': opt_df
    }
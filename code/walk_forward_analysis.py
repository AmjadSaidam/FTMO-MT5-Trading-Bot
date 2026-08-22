import numpy as np 
import pandas as pd

def walk_forward_analysis(data: pd.DataFrame, 
                          data_split_fraction: float = 0.04, 
                          eval_split_fraction: float = 0.01):
    """
    """ 
    # results  
    train_sets: list[pd.DataFrame] = []
    eval_sets: list[pd.DataFrame] = []

    # default vars  
    n = len(data)
    train_len = np.floor(n * data_split_fraction).astype(int)
    eval_len = np.floor(n * eval_split_fraction).astype(int)
    step = np.floor(0.5 * train_len).astype(int)
    number_partions = int(np.floor((n - train_len - eval_len) / step)) + 1

    # return data if 
    if train_len + eval_len > n or step < 1: 
        return data
    
    # otherwise atleast one partion
    left = 0
    for _ in range(number_partions): 
        train_end = left + train_len
        # training
        train_sets.append(
            data.iloc[left: train_end, :].copy() # copy to ensure error, value trying to be set to copy of slice of dataferame
        )
        # evaluation
        eval_sets.append(
            data.iloc[train_end: train_end + eval_len, :].copy()
        )
        left += step

    return {
        'train_sets': train_sets,
        'eval_sets': eval_sets
    }
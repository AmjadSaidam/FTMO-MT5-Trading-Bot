import numpy as np
import pandas as pd

def walk_forward_analysis(data: pd.DataFrame,
                          initial_is: pd.DateOffset = pd.DateOffset(months = 6),
                          oos: pd.DateOffset = pd.DateOffset(months = 1),
                          is_eval_fraction: float = 0.3) -> dict[str, list]:
    """
    anchored (expanding) walk forward folds in calendar time

    - in sample (is): wfa start (data.index[0]) -> is_end, the first fold spans initial_is
    - each in-sample fold is split by bars into model train (first 1 - is_eval_fraction) and in-sample eval (last is_eval_fraction)
    - out of sample (oos): is_end -> is_end + oos, the next fold's is_end is this fold's oos end (step = oos, so oos folds are contiguous
      and every bar after the initial in-sample fold is tested out of sample exactly once)

    returns (end bounds exclusive, sets are slices of data, copy before mutating):
    - train_sets: expanding in-sample folds
    - split_times: first in-sample eval bar per fold, the ml model / grid search history is wfa start -> split_time
    - is_eval_sets: in-sample eval part of each fold (split_time -> is_end), grid search / model selection is scored here
    - eval_sets: out of sample folds
    """
    # results
    train_sets: list[pd.DataFrame] = []
    split_times: list[pd.Timestamp] = []
    is_eval_sets: list[pd.DataFrame] = []
    eval_sets: list[pd.DataFrame] = []

    # default vars
    index = data.index
    is_end = index[0] + initial_is

    while is_end <= index[-1]:
        oos_end = is_end + oos
        is_len = int(index.searchsorted(is_end)) # in-sample bars (wfa start -> is_end)
        oos_end_pos = int(index.searchsorted(oos_end))
        split = int(np.floor(is_len * (1 - is_eval_fraction)))

        if oos_end_pos > is_len and 0 < split < is_len: # skip empty oos periods (eg. no data that month)
            train_sets.append(data.iloc[: is_len])
            split_times.append(index[split])
            is_eval_sets.append(data.iloc[split: is_len])
            eval_sets.append(data.iloc[is_len: oos_end_pos])
        is_end = oos_end # step = oos window

    return {
        'train_sets': train_sets,
        'split_times': split_times,
        'is_eval_sets': is_eval_sets,
        'eval_sets': eval_sets
    }

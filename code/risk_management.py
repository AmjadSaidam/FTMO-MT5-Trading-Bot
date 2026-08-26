import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

# --- DYNAMIC STOP LOSS ---
def atr(data: pd.DataFrame, 
        atr_lookback: int = 14):
    """
    """
    prev_close = data['close'].shift(1) # immediate past price
    true_range = pd.concat([
        data['high'] - data['low'], 
        (data['high'] - prev_close).abs(), 
        (data['low'] - prev_close).abs()
    ], axis = 1).max(axis = 1)
    return true_range.rolling(atr_lookback).mean().to_numpy()

# --- TRADE STOPPING RISK MANAGEMENT ---
def consecutive_loss_threshold(consecutive_loss: int, threshold: int):
    """
    """
    return consecutive_loss >= threshold

# --- ML DRIVEN RISK MANAGEMENT ---
class RandomForestVol():
    """
    classification prediction of loss strategy loss likelihood. Given features of volatility_at_signal and binary strategy_loss label, model will learn to 
    predict if current signal will be winning/lossing. 
    
    to be used in training set of backtest
    """
    def __init__(self, 
                 x_features: pd.DataFrame, 
                 y_label: pd.Series,
                 categorical_col: str):
        valid = y_label.notna() & (x_features[categorical_col].notna()) # only completed trades carry a win/loss label, subset (feature, label) by non None values 
        self.feature_names = x_features.columns
        self.x = x_features[valid] # kept as DataFrame - ColumnTransformer needs column names to select on
        self.y = y_label[valid].to_numpy().astype(int)
        self.categorical_cols = [categorical_col] # list (not bare string) so ColumnTransformer selects a 2D DataFrame slice, not a 1D Series
        self.x_eval = None
        self.y_eval = None
        self.model = None

    def train_model(self, train_split = 0.7):
        """
        """
        min_len = min(self.x.shape[0], self.y.shape[0])
        if min_len < 1:
            return None

        preprocessor = ColumnTransformer(
            transformers = [
                ('cat', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1), self.categorical_cols) # Column Transformer requires keys (name, transformer, columns)
            ], 
            remainder = 'passthrough' # drop other columns with no transformation
        )
        model = Pipeline([
            ('preprocess', preprocessor), 
            ('rf', RandomForestClassifier(class_weight = 'balanced')) # class weight set inversly proportional to label frequency (curbing high class imbalance)
        ])

        if train_split == 1: # no holdout, training set is 100% split 
            self.model = model.fit(self.x, self.y)
            return self.model # encoder set within piepline and automatically set 

        x_train, x_test, y_train, y_test = train_test_split(self.x, 
                                                            self.y, 
                                                            train_size = train_split,
                                                            shuffle = False)
        self.x_eval = x_test
        self.y_eval = y_test
        self.model = model.fit(x_train, y_train)

        return self.model
    
    # write 
    def optimise_train_model(self):
        """optimise model and validate on eval data"""
        return
    
    def model_predict(self,
                      features: dict[str, np.ndarray]):
        """predict using trained model"""
        if self.model is None: # train_model() no-ops when there's not enough labeled in-sample data to fit
            return False
        point_feature_pred = pd.DataFrame(features, columns = self.feature_names) # to ensure features have identical names to trained feature set
        return self.model.predict(point_feature_pred)
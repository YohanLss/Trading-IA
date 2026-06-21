import yfinance as yf
import pandas as pd
import numpy as np
from features import Features


class basePriceModel:
    
    def __init__(self, stock, start , end, horizon,thresholds):
        self.stock = stock
        self.start = start
        self.end = end
        self.horizon = horizon
        self.thresholds = thresholds
        print("horizon =", self.horizon, type(self.horizon))
        print("thresholds =", self.thresholds, type(self.thresholds))
        
    def get_data(self):
        data = yf.download(
            self.stock,
            start=self.start,
            end=self.end,
            auto_adjust=False
        )
        data = data.sort_index()
        # MultiIndex: (Price, Ticker) -> on garde Price uniquement si 1 ticker
        if isinstance(data.columns, pd.MultiIndex):
            # si 1 ticker
            data = data.xs(self.stock, axis=1, level=1)

        return data
    
    #features de bases en fonction de la fenêtre d'horizon
    def get_features(self):
        features_builder = Features(self)
        if self.horizon<10:

            df_feat=features_builder.add_price_short()
        
        
        elif self.horizon >= 10 and self.horizon<=60:

            df_feat=features_builder.add_price_medium()
            
        
        else:
            df_feat=features_builder.add_price_long()
        
        return df_feat
        
    def add_target(self):
        df = self.get_features().copy()
        future_return = df["Close"].shift(-self.horizon) / df["Close"] - 1
        df["target"] =  (future_return >= self.thresholds).astype(int)
        
        return df.dropna()
    
    def time_split(self):
        df = self.add_target().copy()
        n = len(df)

        train = df.iloc[:int(0.6*n)]
        valid = df.iloc[int(0.6*n):int(0.8*n)]
        test  = df.iloc[int(0.8*n):]


        return train, valid, test
    
    def fit(self):


        train, valid, test = self.time_split()
        self.X_train = train.drop(columns=["target"])
        self.X_valid = valid.drop(columns=["target"])
        self.X_test = test.drop(columns=["target"])

        self.y_train = train["target"]
        self.y_valid = valid["target"]
        self.y_test = test["target"]

        return self


if __name__ == "__main__" :
    model = basePriceModel(

    "AAPL",

    "2010-01-01",

    "2025-01-01",

    horizon = 20, thresholds = 0.03

    )

    df = model.add_target()



    
    print(df.head())

    



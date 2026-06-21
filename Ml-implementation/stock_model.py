import yfinance as yf
import pandas as pd
import numpy as np
from features import Features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")





class Stock_model:

    def __init__(self, stock_name, start, end):
        self.stock_name = stock_name
        self.start = start
        self.end = end

    def get_data(self):
        data = yf.download(
            self.stock_name,
            start=self.start,
            end=self.end,
            auto_adjust=False
        )
        data = data.sort_index()
        # MultiIndex: (Price, Ticker) -> on garde Price uniquement si 1 ticker
        if isinstance(data.columns, pd.MultiIndex):
            # si 1 ticker
            data = data.xs(self.stock_name, axis=1, level=1)

        return data

    
    def get_feature_data(self):
        feature_builder = Features(self)
        df_feat = feature_builder.add_features()
        return df_feat
    
    def add_targets(self):
        df = self.get_feature_data().copy()

        sma_50 = df["Close"].rolling(50).mean()
        sma_200 = df["Close"].rolling(200).mean()
        sma_50_slope = sma_50.pct_change(10)
        dist_from_high_50 = df["Close"] / (df["Close"].rolling(50).max() + 1e-9)

        df["target_trending"] = (
            (sma_50 > sma_200) &
            (sma_50_slope > 0) &
            (dist_from_high_50 > 0.95) &
            (df["atr_pct"] < df["atr_pct"].rolling(50).mean() * 1.2)
        ).astype(int)

        df["target_volatile"] = (
            (df["atr_pct"] > df["atr_pct"].rolling(50).mean() * 1.2) &
            (df["volatility_20"] > df["volatility_20"].rolling(50).mean())
        ).astype(int)

        high_20 = df["Close"].rolling(20).max()
        df["target_breakout"] = (
            (df["Close"] >= high_20) &
            (df["volume_spike"] > 1.2)
        ).astype(int)

        df["target_range"] = (
        (df["target_trending"] == 0) &
        (df["target_volatile"] == 0) &
        (df["target_breakout"] == 0)
        ).astype(int)

        return df.dropna()
    
    def time_split(self):
        df = self.add_targets().copy()
        n = len(df)

        train = df.iloc[:int(0.6*n)]
        valid = df.iloc[int(0.6*n):int(0.8*n)]
        test  = df.iloc[int(0.8*n):]


        return train, valid, test
    """
    def fit(self):

        train, valid, test = self.time_split()

        target_colums=["target_trending",
        "target_volatile",
        "target_breakout",
        "target_range"]


        self.X_train = train.drop(columns=target_colums)
        self.X_valid = valid.drop(columns=target_colums)
        self.X_test = test.drop(columns=target_colums)


        self.y_train_trend = train["target_trending"]
        self.y_train_vol = train["target_volatile"]
        self.y_train_break = train["target_breakout"]

        self.y_valid_trend = valid["target_trending"]
        self.y_valid_vol = valid["target_volatile"]
        self.y_valid_break = valid["target_breakout"]

        self.y_test_trend = test["target_trending"]
        self.y_test_vol   = test["target_volatile"]
        self.y_test_break = test["target_breakout"]

        self.model_trending = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        self.model_volatile = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        self.model_breakout = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        self.model_trending.fit(self.X_train, self.y_train_trend)
        self.model_volatile.fit(self.X_train, self.y_train_vol)
        self.model_breakout.fit(self.X_train, self.y_train_break)

        
        print("\n===== VERIFICATION FEATURES =====")

        

        print("target_trending dans X_train ?",
            "target_trending" in self.X_train.columns)

        print("target_volatile dans X_train ?",
            "target_volatile" in self.X_train.columns)

        print("target_breakout dans X_train ?",
            "target_breakout" in self.X_train.columns)

        print("target_range dans X_train ?",
            "target_range" in self.X_train.columns)

        
        

        
        return self
    """
    #entraînement sur le k-fold
    def fit_on_data(
        self,
        X_train,
        y_train_trend,
        y_train_vol,
        y_train_break
    ):

        

        self.model_trending = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        self.model_volatile = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        self.model_breakout = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        self.model_trending.fit(X_train, y_train_trend)
        self.model_volatile.fit(X_train, y_train_vol)
        self.model_breakout.fit(X_train, y_train_break)

        return {
            "trend": self.model_trending,
            "vol": self.model_volatile,
            "break": self.model_breakout
        }
    
    #prédiction sur le k-fold
    def predict_regime_proba_on_data(self, models, X):

        regime_df = pd.DataFrame(index=X.index)

        regime_df["regime_trend_prob"] = models["trend"].predict_proba(X)[:, 1]
        regime_df["regime_vol_prob"] = models["vol"].predict_proba(X)[:, 1]
        regime_df["regime_break_prob"] = models["break"].predict_proba(X)[:, 1]
        

        return regime_df
    
    #tuning oof
    """
    def fit_on_data_with_threshold(self, X_train, y_train_trend, y_train_vol, y_train_break):

        split = int(0.8 * len(X_train))

        set_one = X_train.join(y_train_trend)
        set_two = y_train_vol.join(y_train_break)
        full_set = set_one.join(set_two)

        train_set= full_set.iloc[:split]
        valid_set= full_set.iloc[split:]

        target_colums=["target_trending",
        "target_volatile",
        "target_breakout",
        "target_range"]

        X_train = train_set.drop(colunms=target_colums)
        X_valid =valid_set.drop(columns=target_colums)

        y_t_train=train_set["target_trending"]
        y_v_train=train_set["target_volatile"]
        y_b_train=train_set["target_breakout"]

        y_t_valid=valid_set["target_trending"]
        y_v_valid=valid_set["target_volatile"]
        y_b_valid=valid_set["target_breakout"]

        

        model_trend = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )
        
        model_vol = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        model_break = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
        )

        model_trend.fit(X_train, y_t_train)
        model_vol.fit(X_train, y_v_train)
        model_break.fit(X_train, y_b_train)

        thresholds = np.arange(0.3,0.8,0.05)

        models = {"trending": {"proba":model_trend.predict_proba(X_valid)[:,1],
                               "y":y_t_valid,
                               "best_threshold": 0,
                               "best_score": 0},
                    "volatile": {"proba":model_vol.predict_proba(X_valid)[:,1],
                                 "y":y_v_valid,"best_threshold": 0,
                                 "best_score": 0},
                    "breakout": {"proba":model_break.predict_proba(X_valid)[:,1],
                                 "y":self.y_valid_break,
                                 "best_threshold": 0,
                                 "best_score": 0 }
                }
        
        for name,info in model.items():

            for t in thresholds:
                pred = (info["proba"] >= t).astype(int)

                precision = precision_score(info["y"], pred, zero_division=0)
                recall = recall_score(info["y"], pred, zero_division=0)

                score = precision * recall

                if score > info["best_score"]:
                    info["best_score"] = score
                    best_threshold = t

        

        return {
            "trend": model_trend,
            "vol": model_vol,
            "break": model_break,
            "threshold": best_threshold
        }

            """

        

        

        
    def evaluate_valid(self):

        thresholds = np.arange(0.3,0.8,0.05)

        models = {

            "trending": {

                "proba":

                self.model_trending.predict_proba(self.X_valid)[:,1],

                "y":

                self.y_valid_trend,

                "best_threshold": 0,

                "best_score": 0

            },

            "volatile": {

                "proba":

                self.model_volatile.predict_proba(self.X_valid)[:,1],

                "y":

                self.y_valid_vol,

                "best_threshold": 0,

                "best_score": 0

            },

            "breakout": {

                "proba":

                self.model_breakout.predict_proba(self.X_valid)[:,1],

                "y":

                self.y_valid_break,

                "best_threshold": 0,

                "best_score": 0

            }

        }


        for name, info in models.items():

            print("\n======", name, "======")

            for t in thresholds:

                y_pred = (info["proba"] >= t).astype(int)

                precision = precision_score(
                    info["y"],
                    y_pred
                )

                recall = recall_score(
                    info["y"],
                    y_pred
                )

                score = precision * recall


                print(

                    f"threshold={t:.2f}",

                    f"precision={precision:.3f}",

                    f"recall={recall:.3f}",

                    f"score={score:.3f}"

                )


                if score > info["best_score"]:

                    info["best_score"] = score

                    info["best_threshold"] = t


            print(

                "best threshold",

                info["best_threshold"]

            )


        self.best_thresholds = {

            name: info["best_threshold"]

            for name, info in models.items()

        }

        return self.best_thresholds
    
    

    def evaluate_test(self):

        models = {

            "trending": (
                self.model_trending,
                self.y_test_trend
            ),

            "volatile": (
                self.model_volatile,
                self.y_test_vol
            ),

            "breakout": (
                self.model_breakout,
                self.y_test_break
            )
        }


        for name, (model, y_true) in models.items():

            proba = model.predict_proba(
                self.X_test
            )[:,1]


            threshold = self.best_thresholds[name]


            y_pred = (
                proba >= threshold
            ).astype(int)


            print("\n========== TEST", name, "==========")

            print(

                classification_report(
                    y_true,
                    y_pred
                )

            )


    #pour le cross-validation 
    def predict_regime_proba_on_dataset(self, X):
        regime_df = pd.DataFrame(index=X.index)

        regime_df["regime_trend_prob"] = self.model_trending.predict_proba(X)[:, 1]
        regime_df["regime_vol_prob"] = self.model_volatile.predict_proba(X)[:, 1]
        regime_df["regime_break_prob"] = self.model_breakout.predict_proba(X)[:, 1]

        return regime_df
   
    
    def predict_regime_proba(self, dataset="all"):
        if not hasattr(self, "X_train"):
            raise ValueError("Il faut appeler fit() avant predict_regime_proba().")

        if dataset == "train":
            X = self.X_train
        elif dataset == "valid":
            X = self.X_valid
        elif dataset == "test":
            X = self.X_test
        elif dataset == "all":
            df = self.add_targets().copy()
            target_columns = [
                "target_trending",
                "target_volatile",
                "target_breakout",
                "target_range"
            ]
            X = df.drop(columns=target_columns)
        else:
            raise ValueError("dataset doit être 'train', 'valid', 'test' ou 'all'.")

        regime_df = pd.DataFrame(index=X.index)
        regime_df["regime_trend_prob"] = self.model_trending.predict_proba(X)[:, 1]
        regime_df["regime_vol_prob"] = self.model_volatile.predict_proba(X)[:, 1]
        regime_df["regime_break_prob"] = self.model_breakout.predict_proba(X)[:, 1]

        return regime_df
    
    def evaluate_regime_proba(self, dataset="valid"):
        regime_df = self.predict_regime_proba(dataset=dataset)

        if dataset == "train":
            y_trend = self.y_train_trend
            y_vol = self.y_train_vol
            y_break = self.y_train_break
        elif dataset == "valid":
            y_trend = self.y_valid_trend
            y_vol = self.y_valid_vol
            y_break = self.y_valid_break
        elif dataset == "test":
            y_trend = self.y_test_trend
            y_vol = self.y_test_vol
            y_break = self.y_test_break
        else:
            raise ValueError("dataset doit être 'train', 'valid' ou 'test'.")

        targets = {
            "trending": y_trend,
            "volatile": y_vol,
            "breakout": y_break
        }

        proba_cols = {
            "trending": "regime_trend_prob",
            "volatile": "regime_vol_prob",
            "breakout": "regime_break_prob"
        }

        for name, y_true in targets.items():
            y_proba = regime_df[proba_cols[name]]
            y_pred = (y_proba >= 0.5).astype(int)

            print(f"\n===== {name.upper()} - {dataset} =====")
            print(classification_report(y_true, y_pred))
            print("proba moyenne si y=1 :", y_proba[y_true == 1].mean())
            print("proba moyenne si y=0 :", y_proba[y_true == 0].mean())
        
        

    def predict_current_regime(self):

        df = self.add_targets()

        feature_cols = self.X_train.columns

        latest_row = df.iloc[-1][feature_cols]

        X_latest = latest_row.values.reshape(1,-1)


        result = {
            "trending":

            self.model_trending.predict_proba(X_latest)[0,1],

            "volatile":

            self.model_volatile.predict_proba(X_latest)[0,1],

            "breakout":

            self.model_breakout.predict_proba(X_latest)[0,1]

        }


        return result
        

        

        
if __name__ == "__main__":
    model = Stock_model("PLTR", "2022-01-01", "2026-01-01")
    model.fit()
    model.evaluate_valid()
    model.evaluate_test()
    #print(model.predict_regime_proba("valid").head())
    #model.evaluate_regime_proba("valid")
    #model.evaluate_regime_proba("test")
    
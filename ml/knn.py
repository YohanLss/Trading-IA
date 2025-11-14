import yfinance as yf
import pandas as pd
import talib  # if you have it, for RSI; otherwise you can implement manually
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from services.marketdata import YahooStockMarket

DEFAULT_FEATURE_COLS = [
    "mom_20",
    "mom_60",
    "ema_ratio_10_20",
    "ema_ratio_20_60",
    "dist_ma_20",
    "vol_ratio_5_60",
    "vol_20",
    "rsi_14",
]

ALL_FEATURE_COLS = [
    "5d_return",
    "10d_return",
    "20d_return",
    "60d_return",
    "mom_5",
    "mom_10",
    "mom_20",
    "mom_60",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_60",
    "ema_5",
    "ema_10",
    "ema_20",
    "ema_60",
    "ema_ratio_5_10",
    "ema_ratio_10_20",
    "ema_ratio_5_20",
    "ema_ratio_20_60",
    "price_ratio_5_10",
    "price_ratio_10_20",
    "price_ratio_20_60",
    "price_ratio_5_20",
    "price_ratio_10_60",
    "dist_ma_20",
    "vma_5",
    "vma_10",
    "vma_20",
    "vma_60",
    "vol_ratio_5_10",
    "vol_ratio_10_20",
    "vol_ratio_20_60",
    "vol_ratio_5_20",
    "vol_ratio_5_60",
    "vol_rank_20",
    "ret_1",
    "vol_20",
    "rsi_14",
]


class KNNTrainer:
    """
    Simple wrapper around the KNN pipeline to make it easier to fetch data,
    engineer features, train, and evaluate with a single object.
    """

    def __init__(
            self,
            symbols,
            feature_cols=None,
            target_day=30,
            training_period=365 * 7,
            n_neighbors=41,
            train_frac=0.7,
            model_type="classifier",
            market_data=None,
    ):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.target_day = target_day
        self.training_period = training_period
        self.n_neighbors = n_neighbors
        self.train_frac = train_frac
        self.model_type = model_type
        self.market_data = market_data or YahooStockMarket()

    @staticmethod
    def _normalize_column(df, column):
        return (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    @staticmethod
    def _rolling_rank(x):
        return x.rank().iloc[-1] / len(x)

    def _build_variables(self, stock_history):
        df = stock_history.copy()

        df["5d_return"] = df["Close"].shift(-5) / df["Close"] - 1
        df["10d_return"] = df["Close"].shift(-10) / df["Close"] - 1
        df["20d_return"] = df["Close"].shift(-20) / df["Close"] - 1
        df["60d_return"] = df["Close"].shift(-60) / df["Close"] - 1

        df["mom_5"] = df["Close"] / df["Close"].shift(5) - 1
        df["mom_10"] = df["Close"] / df["Close"].shift(10) - 1
        df["mom_20"] = df["Close"] / df["Close"].shift(20) - 1
        df["mom_60"] = df["Close"] / df["Close"].shift(60) - 1

        df["sma_5"] = df["Close"].rolling(5).mean()
        df["sma_10"] = df["Close"].rolling(10).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["sma_60"] = df["Close"].rolling(60).mean()

        df["ema_5"] = df["Close"].ewm(span=5, adjust=False).mean()
        df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema_60"] = df["Close"].ewm(span=60, adjust=False).mean()

        df["ema_ratio_5_10"] = df["ema_5"] / df["ema_10"]
        df["ema_ratio_10_20"] = df["ema_10"] / df["ema_20"]
        df["ema_ratio_5_20"] = df["ema_5"] / df["ema_20"]
        df["ema_ratio_20_60"] = df["ema_20"] / df["ema_60"]

        df["price_ratio_5_10"] = df["sma_5"] / df["sma_10"]
        df["price_ratio_10_20"] = df["sma_10"] / df["sma_20"]
        df["price_ratio_20_60"] = df["sma_20"] / df["sma_60"]

        df["price_ratio_5_20"] = df["sma_5"] / df["sma_20"]
        df["price_ratio_10_60"] = df["sma_10"] / df["sma_60"]
        df["dist_ma_20"] = (df["Close"] - df["sma_20"]) / df["sma_20"]

        df["vma_5"] = df["Volume"].rolling(5).mean()
        df["vma_10"] = df["Volume"].rolling(10).mean()
        df["vma_20"] = df["Volume"].rolling(20).mean()
        df["vma_60"] = df["Volume"].rolling(60).mean()

        df["vol_ratio_5_10"] = df["vma_5"] / df["vma_10"]
        df["vol_ratio_10_20"] = df["vma_10"] / df["vma_20"]
        df["vol_ratio_20_60"] = df["vma_20"] / df["vma_60"]
        df["vol_ratio_5_20"] = df["vma_5"] / df["vma_20"]
        df["vol_ratio_5_60"] = df["vma_5"] / df["vma_60"]

        df["vol_rank_20"] = df["Volume"].rolling(20).apply(self._rolling_rank, raw=False)

        df["ret_1"] = df["Close"].pct_change()
        df["vol_20"] = df["ret_1"].rolling(20).std()

        df["rsi_14"] = talib.RSI(df["Close"].values, timeperiod=14)
        df["rsi_14"] = self._normalize_column(df, "rsi_14")

        T = self.target_day
        df["future_return_T"] = df["Close"].shift(-T) / df["Close"] - 1
        df["future_up"] = (df["Close"].shift(-T) > df["Close"]).astype(int)

        return df[self.feature_cols + ["future_return_T", "future_up"]].dropna()

    def _compile_feature_set(self):
        frames = []
        for symbol in self.symbols:
            history = self.market_data.get_stock_history(
                symbol=symbol, days=self.training_period, return_df=True
            )
            frames.append(self._build_variables(history))
        if not frames:
            raise ValueError("No symbols provided for training.")
        return pd.concat(frames, ignore_index=True)

    def _time_series_split(self, X, y):
        split_idx = int(len(X) * self.train_frac)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def _train_classifier(self, dataset):
        X = dataset[self.feature_cols].values
        y = dataset["future_up"].values

        X_train, X_test, y_train, y_test = self._time_series_split(X, y)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=self.n_neighbors)),
            ]
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "label_distribution": np.bincount(y_test),
            "classification_report": classification_report(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
        }
        return model, metrics

    def _train_regressor(self, dataset):
        X = dataset[self.feature_cols].values
        y = dataset["future_return_T"].values

        X_train, X_test, y_train, y_test = self._time_series_split(X, y)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "knn",
                    KNeighborsRegressor(
                        n_neighbors=self.n_neighbors,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
        }
        return model, metrics

    def train(self):
        dataset = self._compile_feature_set()
        if self.model_type == "classifier":
            model, metrics = self._train_classifier(dataset)
        elif self.model_type == "regressor":
            model, metrics = self._train_regressor(dataset)
        else:
            raise ValueError("model_type must be 'classifier' or 'regressor'")

        return model, metrics


def main():
    config1 = {
        "symbols": ["AVGO"],
        "model_type": "classifier",
        "n_neighbors": 41,
        "feature_cols": [
            "mom_20",
            "mom_60",
            "ema_ratio_10_20",
            "ema_ratio_20_60",
            "dist_ma_20",
            "vol_ratio_5_60",
            "vol_20",
            "rsi_14",
        ]
    }

    # Config 2: multi symbol classifier, shorter horizon, more recent data
    # Idea: more symbols, smaller k, T = 10
    config2 = {
        "symbols": ["AVGO"],
        "model_type": "classifier",
        "n_neighbors": 21,
        "training_period": 365 * 3,
        "target_day": 10,
        "feature_cols": [
            "mom_10",
            "mom_20",
            "ema_ratio_5_10",
            "ema_ratio_10_20",
            "vol_ratio_5_20",
            "vol_rank_20",
            "vol_20",
            "rsi_14",
        ],
    }

    # Config 3: return based classifier, medium horizon with longer history
    # This uses only backward looking returns and a few trend features
    config3 = {
        "symbols": ["AVGO", "NVDA", "TSLA", "AMZN", "GOOGL"],
        "model_type": "classifier",
        "n_neighbors": 31,
        "training_period": 365 * 5,
        "target_day": 20,
        "feature_cols": [
            "5d_return",
            "10d_return",
            "20d_return",
            "60d_return",
            "price_ratio_5_10",
            "price_ratio_10_20",
            "price_ratio_20_60",
            "vol_ratio_5_20",
            "vol_rank_20",
        ],
    }

    # Config 4: regression model, medium horizon, multi symbol
    # Predict numeric future_return_T instead of direction
    config4 = {
        "symbols": ["AVGO"],
        "model_type": "regressor",
        "n_neighbors": 51,
        "training_period": 365 * 5,
        "target_day": 20,
        "feature_cols": [
            "5d_return",
            "10d_return",
            "20d_return",
            "mom_10",
            "mom_20",
            "ema_ratio_5_10",
            "ema_ratio_10_20",
            "vol_20",
            "rsi_14",
        ],
    }

    # Config 5: long horizon trend follower, single strong stock
    # T = 60 days, longer training, slow features
    config5 = {
        "symbols": ["AVGO"],
        "model_type": "classifier",
        "n_neighbors": 41,
        "training_period": 365 * 10,
        "target_day": 60,
        "feature_cols": [
            "mom_20",
            "mom_60",
            "ema_ratio_10_20",
            "ema_ratio_20_60",
            "dist_ma_20",
            "vol_ratio_5_60",
            "vol_20",
            "rsi_14",
        ],
    }
    
    config = config5
    trainer = KNNTrainer(**config)
    model, metrics = trainer.train()

    print("KNN model trained.")
    if trainer.model_type == "classifier":
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Label distribution:", metrics["label_distribution"])
        print(metrics["classification_report"])
    else:
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"R^2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()

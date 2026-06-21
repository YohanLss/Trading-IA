
import pandas as pd
import numpy as np

class Features:

    def __init__(self,stock_model):
        self.stock_model = stock_model

    def add_features(self):
        df = self.stock_model.get_data().copy()
        df = self._add_trend_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_breakout_features(df)
        df = self._add_interaction_features(df)
        return df
    
    #Features pour le price_model : 
    #1) common features
    def add_features_stock(self):
        df = self.stock_model.get_data().copy()
        df = self._add_trend_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_breakout_features(df)
        df = self._add_interaction_features(df)
        return df

    #contexte_générale
    #court terme
    def add_price_short(self):
        df = self.stock_model.get_data().copy()

        eps = 1e-9
        # RETURNS COURT TERME
        df["return_5d"] = df["Close"].pct_change(5)

        # ACCELERATION
        df["price_acceleration"] = df["return_5d"].diff()

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / (avg_loss + eps)

        df["RSI"] = 100 - (100 / (1 + rs))

        # VOLATILITE COURT TERME
        returns = df["Close"].pct_change()

        df["volatility_20"] = returns.rolling(20).std()

        # ATR
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()

        df["atr_pct"] = atr / (df["Close"] + eps)

        # VOLUME RECENT
        df["volume_change"] = df["Volume"].pct_change()

        volume_ma_20 = df["Volume"].rolling(20).mean()

        df["volume_spike"] = df["Volume"] / (volume_ma_20 + eps)

        

        # MOMENTUM CONFIRME PAR VOLUME
        df["momentum_volume"] = (
            df["return_5d"] *
            df["volume_spike"]
        )

        # COMPRESSION VOLATILITE
        atr_mean_50 = df["atr_pct"].rolling(50).mean()

        df["volatility_compression"] = (
            df["atr_pct"] /
            (atr_mean_50 + eps)
        )

        return df

        # CONTEXTE REGIME
        # ajoutés après StockModel
        # regime_trend_prob
        # regime_vol_prob
        # regime_break_prob

        #moyen terme
    def add_price_medium(self):
        df = self.stock_model.get_data().copy()

        eps = 1e-9
        # RETURNS MOYEN TERME
        df["return_20d"] = df["Close"].pct_change(20)

        df["return_60d"] = df["Close"].pct_change(60)

        # RATIO MOMENTUM
        df["momentum_ratio_20_60"] = (
            df["return_20d"] /
            (df["return_60d"] + eps)
        )

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / (avg_loss + eps)

        df["RSI"] = 100 - (100 / (1 + rs))

        # MOYENNES MOBILES
        sma_50 = df["Close"].rolling(50).mean()

        df["dist_sma_50"] = (
            (df["Close"] - sma_50) /
            (sma_50 + eps)
        )

        df["sma_50_slope"] = sma_50.pct_change(10)

        # TREND GLOBAL
        sma_200 = df["Close"].rolling(200).mean()

        df["trend_strength"] = (
            sma_50 /
            (sma_200 + eps)
        )

        df["golden_cross"] = (
            sma_50 > sma_200
        ).astype(int)

        # POSITION DANS LA TENDANCE
        high_50 = df["Close"].rolling(50).max()

        df["dist_from_high_50"] = (
            df["Close"] /
            (high_50 + eps)
        )

        # VOLATILITE
        returns = df["Close"].pct_change()

        df["volatility_20"] = returns.rolling(20).std()

        # ATR
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()

        df["atr_pct"] = atr / (df["Close"] + eps)

        # VOLUME
        volume_ma_20 = df["Volume"].rolling(20).mean()

        volume_ma_60 = df["Volume"].rolling(60).mean()

        df["volume_spike"] = df["Volume"] / (volume_ma_20 + eps)

        df["volume_regime"] = volume_ma_20 / (volume_ma_60 + eps)

        # INTERACTIONS IMPORTANTES
        df["momentum_volume"] = (
            df["return_20d"] *
            df["volume_spike"]
        )

        df["trend_volatility"] = (
            df["trend_strength"] *
            df["atr_pct"]
        )

        return df
    
    def add_price_long(self):
        df = self.stock_model.get_data().copy()

        eps = 1e-9
        # RETURNS LONG TERME

        df["return_60d"] = df["Close"].pct_change(60)

        # MOYENNES LONG TERME
        sma_200 = df["Close"].rolling(200).mean()

        df["dist_sma_200"] = (
            (df["Close"] - sma_200) /
            (sma_200 + eps)
        )

        df["sma_200_slope"] = sma_200.pct_change(20)

        # TREND GLOBAL
        sma_50 = df["Close"].rolling(50).mean()

        df["trend_strength"] = (
            sma_50 /
            (sma_200 + eps)
        )

        df["golden_cross"] = (
            sma_50 > sma_200
        ).astype(int)

        # STABILITE TENDANCE
        returns = df["Close"].pct_change()

        df["trend_consistency_50"] = (
            (returns > 0)
            .rolling(50)
            .mean()
        )

        # POSITION DANS LE CYCLE
        high_200 = df["Close"].rolling(200).max()

        df["dist_from_high_200"] = (
            df["Close"] /
            (high_200 + eps)
        )

        # VOLATILITE LONG TERME
        df["volatility_60"] = returns.rolling(60).std()

        # ATR
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()

        df["atr_pct"] = atr / (df["Close"] + eps)

        # VOLUME LONG TERME
        volume_ma_60 = df["Volume"].rolling(60).mean()

        volume_ma_120 = df["Volume"].rolling(120).mean()

        df["volume_regime"] = (
            volume_ma_60 /
            (volume_ma_120 + eps)
        )

        # INTERACTION STRUCTURELLE
        df["trend_volatility"] = (
            df["trend_strength"] *
            df["atr_pct"]
        )

        return df

    def _add_trend_features(self, df):
        #peut-être plus les utiliser comme features plutôt
        #que variable

        sma_20 = df["Close"].rolling(20).mean()
        sma_50 = df["Close"].rolling(50).mean()
        df["sma_100"] = df["Close"].rolling(100).mean()
        sma_200 = df["Close"].rolling(200).mean()

        ema_10 = df["Close"].ewm(span=10).mean()
        ema_20 = df["Close"].ewm(span=20).mean()
        ema_50 = df["Close"].ewm(span=50).mean()

        #plus important
        #moyenne des tendances
        df["sma_ratio_20_50"] = sma_20 / (sma_50 + 1e-9)
        df["sma_ratio_50_200"] = sma_50 / (sma_200 + 1e-9)

        df["ema_ratio_10_20"] = ema_10 / (ema_20 + 1e-9)
        df["ema_ratio_20_50"] = ema_20 / (ema_50 + 1e-9)

        #position par rapport à la tendance(higher,lower)
        df["dist_sma_20"] = (df["Close"] - sma_20) / (sma_20 + 1e-9)
        df["dist_sma_50"] = (df["Close"] - sma_50) / (sma_50 + 1e-9)
        df["dist_sma_200"] = (df["Close"] - sma_200) / (sma_200 + 1e-9)

        # pente des moyennes
        df["sma_20_slope"] = sma_20.pct_change(5)
        df["sma_50_slope"] = sma_50.pct_change(10)
        df["sma_200_slope"] = sma_200.pct_change(20)

        #force de la tendance
        df["golden_cross"] = (sma_50 > sma_200).astype(int)
        df["trend_strength"] = sma_50 / (sma_200 + 1e-9)
        df["trend_direction"] = (df["sma_50_slope"] > 0).astype(int)

        #stabilité de la tendance
        returns = df["Close"].pct_change()
        df["trend_consistency_20"] = (returns > 0).rolling(20).mean()
        df["trend_consistency_50"] = (returns > 0).rolling(50).mean()

        #position dans la tendance
        df["dist_from_high_20"] = df["Close"] / (df["Close"].rolling(20).max() + 1e-9)
        df["dist_from_high_50"] = df["Close"] / (df["Close"].rolling(50).max() + 1e-9)
        df["dist_from_high_200"] = df["Close"] / (df["Close"].rolling(200).max() + 1e-9)

        return df

    def _add_momentum_features(self, df):
        eps = 1e-9

        # returns sur plusieurs horizons
        
        df["return_20d"] = df["Close"].pct_change(20)
        df["return_60d"] = df["Close"].pct_change(60)

        # ratio de momentum
        df["momentum_ratio_20_60"] = df["return_20d"] / (df["return_60d"] + eps)

        # accélération du mouvement
        df["price_acceleration"] = df["return_20d"].diff()

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / (avg_loss + eps)
        df["RSI"] = 100 - (100 / (1 + rs))

        return df
        

    def _add_volatility_features(self, df):
        eps = 1e-9

        ret = df["Close"].pct_change()

        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        # volatilité réalisée
        
        df["volatility_20"] = ret.rolling(20).std()
        df["volatility_60"] = ret.rolling(60).std()

        # True Range / ATR
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df["atr_pct"] = atr / (df["Close"] + eps)

        # régime de volatilité
        atr_mean_50 = df["atr_pct"].rolling(50).mean()
        df["high_volatility_regime"] = (df["atr_pct"] > atr_mean_50).astype(int)

        # compression / expansion
        df["volatility_compression"] = df["atr_pct"] / (atr_mean_50 + eps)

        return df

    def _add_volume_features(self, df):
        eps = 1e-9

        df["volume_change"] = df["Volume"].pct_change()

        
        volume_ma_20 = df["Volume"].rolling(20).mean()
        volume_ma_60 = df["Volume"].rolling(60).mean()

        df["volume_spike"] = df["Volume"] / (volume_ma_20 + eps)
        df["volume_regime"] = volume_ma_20 / (volume_ma_60 + eps)

        

        return df
        

    def _add_breakout_features(self, df):

        eps = 1e-9

        # distance par rapport aux plus hauts récents
        high_20 = df["Close"].rolling(20).max()
        high_50 = df["Close"].rolling(50).max()

        # indicateur de nouveau sommet
        df["new_high_20"] = (df["Close"] >= high_20).astype(int)
        df["new_high_50"] = (df["Close"] >= high_50).astype(int)

        return df

    def _add_interaction_features(self, df):

        eps = 1e-9

        # momentum + volume
        df["momentum_volume"] = df["return_20d"] * df["volume_spike"]

        # trend + volatilité
        df["trend_volatility"] = df["trend_strength"] * df["atr_pct"]


        return df

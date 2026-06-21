import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score
from tendence_model import TrendPriceModel
from volatile_model import VolatileModel
from breakout_model import BreakoutModel
from stock_model import Stock_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

class MetaModel:

    def __init__(self, tendence_model, volatile_model, breakout_model, stock_model,thresholds,horizon):
        self.tendence_model = tendence_model
        self.volatile_model = volatile_model
        self.breakout_model = breakout_model
        self.stock_model = stock_model
        self.thresholds = thresholds
        self.horizon = horizon

    def build_meta_features(self, split="valid"):

        self.tendence_model._ensure_ready()
        self.volatile_model._ensure_ready()
        self.breakout_model._ensure_ready()
        #le dataset du meta se fait sur le validation set ou test set

        if split == "valid":
            trend_pred = self.tendence_model.predict_history(X=self.tendence_model.X_valid)
            vol_pred = self.volatile_model.predict_history(X=self.volatile_model.X_valid)
            break_pred = self.breakout_model.predict_history(X=self.breakout_model.X_valid)

        elif split == "test":
            trend_pred = self.tendence_model.predict_history(X=self.tendence_model.X_test)
            vol_pred = self.volatile_model.predict_history(X=self.volatile_model.X_test)
            break_pred = self.breakout_model.predict_history(X=self.breakout_model.X_test)

        else:
            raise ValueError("split doit être 'valid' ou 'test'")

        if not trend_pred.index.equals(vol_pred.index):
            raise ValueError("Les index trend et volatile ne sont pas alignés.")

        if not trend_pred.index.equals(break_pred.index):
            raise ValueError("Les index trend et breakout ne sont pas alignés.")

        regime_full = self.stock_model.predict_regime_proba("all")
        regime_pred = regime_full.loc[trend_pred.index]

        

        df = pd.concat([
            trend_pred[["trend_up_proba"]],
            vol_pred[["vol_up_proba"]],
            break_pred[["break_up_proba"]],
            regime_pred
        ], axis=1)

        df["expert_agreement"] = (
            df[[
                "trend_up_proba",
                "break_up_proba",
                "vol_up_proba"
            ]].std(axis=1)
        )

        df["trend_break_alignment"] = (
            df["trend_up_proba"]
            * df["break_up_proba"]
        )

        df["trend_regime_alignment"] = (
            df["trend_up_proba"]
            * df["regime_trend_prob"]
        )

        df["break_regime_alignment"] = (
            df["break_up_proba"]
            * df["regime_break_prob"]
        )

        df["volatility_penalty"] = (
            df["vol_up_proba"]
            * (1 - df["regime_trend_prob"])
        )

        print(df.columns.tolist())
        print(df.head())

        return df

    def add_target(self, split="valid"):

        df = self.build_meta_features(split=split).copy()

        price_df = self.tendence_model.get_features().copy()

        future_return = (
            price_df["Close"].shift(-self.horizon) / price_df["Close"] - 1
        )

        df["future_return"] = future_return.reindex(df.index)
        df["target"] = (df["future_return"] > self.thresholds).astype(int)

        df = df.dropna().copy()

        return df
    

    #for the crossvalidation method
    def build_cv_meta_model(self):
        
        # datasets complets de chaque expert
        df_stock = self.stock_model.add_targets().copy()
        df_trend = self.tendence_model.add_target().copy()
        df_vol = self.volatile_model.add_target().copy()
        df_break = self.breakout_model.add_target().copy()

        # garder uniquement l'index commun
        # s'assure de l'alignement du dataset
        common_index = (
            df_trend.index
            .intersection(df_vol.index)
            .intersection(df_break.index)
            .intersection(df_stock.index)
        )

        stock_target_cols = [
            "target_trending",
            "target_volatile",
            "target_breakout",
            "target_range"
        ]

        
        

        #dataset bien aligné
        df_trend = df_trend.loc[common_index].copy()
        df_vol = df_vol.loc[common_index].copy()
        df_break = df_break.loc[common_index].copy()
        df_stock = df_stock.loc[common_index].copy()

        price_df = self.tendence_model.get_features().loc[common_index]

        #expert components
        X_trend = df_trend.drop(columns=["target", "future_return"], errors="ignore")
        y_trend = df_trend["target"]

        X_vol = df_vol.drop(columns=["target", "future_return"], errors="ignore")
        y_vol = df_vol["target"]

        X_break = df_break.drop(columns=["target", "future_return"], errors="ignore")
        y_break = df_break["target"]

        X_stock = df_stock.drop(columns=stock_target_cols, errors="ignore")
        y_stock_trend = df_stock["target_trending"]
        y_stock_vol = df_stock["target_volatile"]
        y_stock_break = df_stock["target_breakout"]

        print((y_trend == y_vol).all())
        print((y_trend == y_break).all())
        # target finale du meta-model
        y_meta = y_trend.copy()

          

        trend_oof_pred = np.full(len(common_index), np.nan)
        vol_oof_pred = np.full(len(common_index), np.nan)
        break_oof_pred = np.full(len(common_index), np.nan)


        regime_trend_oof = np.full(len(common_index), np.nan)
        regime_vol_oof = np.full(len(common_index), np.nan)
        regime_break_oof = np.full(len(common_index), np.nan)


        trend_oof_signal = np.full(len(common_index), np.nan)
        vol_oof_signal = np.full(len(common_index), np.nan)
        break_oof_signal = np.full(len(common_index), np.nan)

        tscv = TimeSeriesSplit(
            n_splits=5,
            test_size=200,
            max_train_size=800
        )

        for train_idx, valid_idx in tscv.split(X_trend):
            # folds trend
            X_train_trend = X_trend.iloc[train_idx]
            y_train_trend = y_trend.iloc[train_idx]
            X_valid_trend = X_trend.iloc[valid_idx]

            # folds vol
            X_train_vol = X_vol.iloc[train_idx]
            y_train_vol = y_vol.iloc[train_idx]
            X_valid_vol = X_vol.iloc[valid_idx]

            # folds breakout
            X_train_break = X_break.iloc[train_idx]
            y_train_break = y_break.iloc[train_idx]
            X_valid_break = X_break.iloc[valid_idx]

            #folds stocks
            X_train_stock = X_stock.iloc[train_idx]
            y_train_stockTend = y_stock_trend.iloc[train_idx]
            y_train_stockVol = y_stock_vol.iloc[train_idx]
            y_train_stockBreak = y_stock_break.iloc[train_idx]
            X_valid_stock = X_stock.iloc[valid_idx]

            #rajout des régimes
            stock_models_fold = self.stock_model.fit_on_data(
                X_train_stock,
                y_train_stockTend,
                y_train_stockVol,
                y_train_stockBreak
            )

            regime_train = self.stock_model.predict_regime_proba_on_data(
                stock_models_fold,
                X_train_stock
            )

            regime_valid = self.stock_model.predict_regime_proba_on_data(
                stock_models_fold,
                X_valid_stock
            )

            regime_trend_oof[valid_idx] = regime_valid["regime_trend_prob"]
            regime_vol_oof[valid_idx] = regime_valid["regime_vol_prob"]
            regime_break_oof[valid_idx] = regime_valid["regime_break_prob"]


            X_train_trend = X_train_trend.join(regime_train)
            X_valid_trend = X_valid_trend.join(regime_valid)

            X_train_vol = X_train_vol.join(regime_train)
            X_valid_vol = X_valid_vol.join(regime_valid)

            X_train_break = X_train_break.join(regime_train)
            X_valid_break = X_valid_break.join(regime_valid)


            # fit experts
            #print(f"Colonnes du fold tendence sur train: {X_train_trend.columns}")
            #print(f"Colonnes du fold tendence sur valid: {X_valid_trend.columns}")
            trend_model_fold,trend_threshold = self.tendence_model.fit_on_data_with_threshold(X_train_trend, y_train_trend)
            vol_model_fold,vol_threshold = self.volatile_model.fit_on_data_with_threshold(X_train_vol, y_train_vol)
            break_model_fold,break_threshold = self.breakout_model.fit_on_data_with_threshold(X_train_break, y_train_break)

            trend_proba = self.tendence_model.predict_proba_on_data(trend_model_fold,X_valid_trend)
            vol_proba = self.volatile_model.predict_proba_on_data(vol_model_fold,X_valid_vol)
            break_proba = self.breakout_model.predict_proba_on_data(break_model_fold,X_valid_break)

            # OOF predictions avec tuning
            

            
            

            trend_oof_pred[valid_idx] = trend_proba
            trend_oof_signal[valid_idx] = (trend_proba >= trend_threshold).astype(int)

            vol_oof_pred[valid_idx]= vol_proba
            vol_oof_signal[valid_idx] = (vol_proba >= vol_threshold).astype(int)

            break_oof_pred[valid_idx]= break_proba
            break_oof_signal[valid_idx] = (break_proba >= break_threshold).astype(int)

        #regime_full = self.stock_model.predict_regime_proba("all").loc[common_index]

        meta_df = pd.DataFrame({
            "trend_up_proba": trend_oof_pred,
            "vol_up_proba": vol_oof_pred,
            "break_up_proba": break_oof_pred,
            "regime_trend_prob": regime_trend_oof,
            "regime_vol_prob": regime_vol_oof,
            "regime_break_prob": regime_break_oof,
            #"trend_signal": trend_oof_signal,
            #"vol_signal": vol_oof_signal,
            #"break_signal": break_oof_signal
        }, index=common_index)
        #meta_df = meta_df.join(regime_full)

        # =========================
        # REAL MARKET FEATURES
        # =========================

        # Volatilité réelle
        
        meta_df["volatility_20"] = price_df["volatility_20"]
        #meta_df["atr_pct"] = price_df["atr_pct"]

        # Volume réel
        #meta_df["volume_spike"] = price_df["volume_spike"]
        meta_df["volume_ratio"] = price_df["volume_regime"]

        # Sentiment proxy réel
        #meta_df["rsi"] = price_df["RSI"]
        meta_df["dist_from_high"] = price_df["dist_from_high_50"]

        # Momentum / direction enrichie
        meta_df["return_20d"] = price_df["return_20d"]
        meta_df["sma_slope"] = price_df["sma_50_slope"]
        
        meta_df["expert_std"] = meta_df[
            ["trend_up_proba", "vol_up_proba", "break_up_proba"]
        ].std(axis=1)

        meta_df["expert_range"] = (
            meta_df[[
                "trend_up_proba",
                "vol_up_proba",
                "break_up_proba"
            ]].max(axis=1)
            -
            meta_df[[
                "trend_up_proba",
                "vol_up_proba",
                "break_up_proba"
            ]].min(axis=1)
        )

        
                        
        

        """
        meta_df["expert_mean"] = meta_df[
            ["trend_up_proba", "vol_up_proba", "break_up_proba"]
        ].mean(axis=1)

        meta_df["expert_min"] = meta_df[
            ["trend_up_proba", "vol_up_proba", "break_up_proba"]
        ].min(axis=1)
        """

        meta_df["trend_expert_weighted"] = (
            meta_df["trend_up_proba"] * meta_df["regime_trend_prob"]
        )

        meta_df["vol_expert_weighted"] = (
            meta_df["vol_up_proba"] * meta_df["regime_vol_prob"]
        )

        meta_df["break_expert_weighted"] = (
            meta_df["break_up_proba"] * meta_df["regime_break_prob"]
        )

        meta_df["best_expert_proba"] = meta_df[[
            "trend_expert_weighted",
            "vol_expert_weighted",
            "break_expert_weighted"
        ]].max(axis=1)

        meta_df["volatility_penalty"] = (
            meta_df["vol_up_proba"]
            * (1 - meta_df["regime_trend_prob"])
        )

        meta_df["trend_efficiency"] = (
            meta_df["return_20d"]
            / (meta_df["volatility_20"] + 1e-6)

        )

        meta_df["best_expert_gap"] = (
            meta_df["best_expert_proba"]
            -
            meta_df[[
                "trend_expert_weighted",
                "vol_expert_weighted",
                "break_expert_weighted"
            ]].mean(axis=1)
        )

        

        meta_df["target"] = y_meta
        print(meta_df.shape)
        meta_df = meta_df.dropna().copy()
        self.df_meta = meta_df

        #il ne manque plus que les sentiment analysis features pour compléter le modèle
        
        return meta_df




    def train_meta_model(self):
            
            if not hasattr(self, "df_meta"):
                self.build_cv_meta_model()

   

            self.meta_horizon = self.horizon
            self.meta_target_threshold = self.thresholds

            df_meta = self.df_meta
            df_meta = df_meta.sort_index()

            n = len(df_meta)

            train_end = int(0.6 * n)
            valid_end = int(0.8 * n)

            meta_train = df_meta.iloc[:train_end]
            meta_valid = df_meta.iloc[train_end:valid_end]
            meta_test  = df_meta.iloc[valid_end:]

            self.X_meta_train = meta_train.drop(columns=["target"])
            self.y_meta_train = meta_train["target"]

            self.X_meta_valid = meta_valid.drop(columns=["target"])
            self.y_meta_valid = meta_valid["target"]

            self.X_meta_test = meta_test.drop(columns=["target"])
            self.y_meta_test = meta_test["target"]

            scale_pos_weight = (
                    len(self.y_meta_train) - self.y_meta_train.sum()
                ) / self.y_meta_train.sum()
            
            base_model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2,
                random_state=42,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight
            )
            """
            base_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42
            )
            """

            self.model = base_model


            self.model.fit(self.X_meta_train,self.y_meta_train)

            # ===== FEATURE IMPORTANCE =====
            
            base_fitted = self.model

            importance = pd.Series(
                base_fitted.feature_importances_,
                index=self.X_meta_train.columns
            ).sort_values(ascending=False)

            print("\n===== FEATURE IMPORTANCE DANS META =====")
            print(importance.head(20))

            
            return self

    def evaluate_meta_model(self, threshold=None):

        from sklearn.metrics import (
            classification_report,
            precision_score,
            recall_score,
            f1_score,
            balanced_accuracy_score
        )

        if not hasattr(self, "model"):
            self.train_meta_model()

        X_valid = self.X_meta_valid
        y_valid = self.y_meta_valid

        X_test = self.X_meta_test
        y_test = self.y_meta_test

        proba_valid = self.model.predict_proba(X_valid)[:, 1]

        if threshold is None:
            thresholds = np.arange(0.05, 0.91, 0.05)

            best_score = -1
            best_threshold = 0.50

            print("\n===== THRESHOLD SEARCH ON VALID =====")

            for t in thresholds:
                y_pred_valid = (proba_valid >= t).astype(int)

                precision = precision_score(y_valid, y_pred_valid, zero_division=0)
                recall = recall_score(y_valid, y_pred_valid, zero_division=0)
                f1 = f1_score(y_valid, y_pred_valid, zero_division=0)
                bal_acc = balanced_accuracy_score(y_valid, y_pred_valid)
                signal_rate = y_pred_valid.mean()

                # score simple et stable
                target_signal_rate = 0.40

                #score = f1 * (1 - abs(signal_rate - target_signal_rate))
                score = (
                    0.45 * f1
                    + 0.35 * precision
                    + 0.20 * bal_acc
                )
                print(f"\nBest VALID threshold = {best_threshold:.2f} | score={best_score:.3f}")

                print(
                    f"threshold={t:.2f} | "
                    f"precision={precision:.3f} | "
                    f"recall={recall:.3f} | "
                    f"f1={f1:.3f} | "
                    f"bal_acc={bal_acc:.3f} | "
                    f"signal_rate={signal_rate:.3f} | "
                    f"score={score:.3f}"
                )

                if score > best_score:
                    best_score = score
                    best_threshold = t

            self.meta_best_threshold = best_threshold
        

        else:
            self.meta_best_threshold = threshold

        

        proba_test = self.model.predict_proba(X_test)[:, 1]

        confidence = proba_test

        # filtre marché propre
        trend_setup = (
            (X_test["trend_up_proba"] >= 0.60)
            & (X_test["regime_trend_prob"] >= 0.50)
            & (X_test["regime_vol_prob"] <= 0.60)
        )

        breakout_setup = (
            (X_test["break_up_proba"] >= 0.60)
            & (X_test["regime_break_prob"] >= 0.45)
            & (X_test["trend_up_proba"] >= 0.50)
        )

        volatility_setup = (
            (X_test["vol_up_proba"] >= 0.60)
            & (X_test["regime_vol_prob"] >= 0.50)
            & (X_test["trend_up_proba"] >= 0.55)
        )
        """

        #je teste plusieurs algorithmes de filtrage
        #premier
        
        # filtre confiance
        setup_score = (
            0.35 * X_test["trend_up_proba"]
            + 0.25 * X_test["break_up_proba"]
            + 0.15 * X_test["vol_up_proba"]
            + 0.15 * X_test["regime_trend_prob"]
            + 0.10 * X_test["regime_break_prob"]
            - 0.15 * X_test["regime_vol_prob"]
        )

        setup_score_norm = (
            (setup_score - setup_score.mean())
            / (setup_score.std() + 1e-6)
        )

        target_signal_rate = 0.25  # 20–30% idéal


        combined_score = proba_test * (setup_score_norm + 2)

        threshold = np.quantile(combined_score, 0.75)

        y_pred = (combined_score >= threshold).astype(int)
        
        #deuxième
        setup_score = (
            0.35 * X_test["trend_up_proba"]
            + 0.25 * X_test["break_up_proba"]
            + 0.15 * X_test["vol_up_proba"]
            + 0.15 * X_test["regime_trend_prob"]
            + 0.10 * X_test["regime_break_prob"]
            - 0.15 * X_test["regime_vol_prob"]
        )

        setup_score_norm = (
            (setup_score - setup_score.mean())
            / (setup_score.std() + 1e-6)
        )
        setup_bonus = (
            0.2 * trend_setup.astype(int)
            + 0.2 * breakout_setup.astype(int)
            + 0.1 * volatility_setup.astype(int)
        )

        combined_score = proba_test * (setup_score_norm + 2 + setup_bonus)

        threshold = np.quantile(combined_score, 0.75)

        y_pred = (combined_score >= threshold).astype(int)
        
        #troisième
        setup_score = (
            0.35 * X_test["trend_up_proba"]
            + 0.25 * X_test["break_up_proba"]
            + 0.15 * X_test["vol_up_proba"]
            + 0.15 * X_test["regime_trend_prob"]
            + 0.10 * X_test["regime_break_prob"]
            - 0.15 * X_test["regime_vol_prob"]
        )

        setup_score_norm = (
            (setup_score - setup_score.mean())
            / (setup_score.std() + 1e-6)
        )
        setup_strength = (
            trend_setup.astype(int)
            + breakout_setup.astype(int)
            + volatility_setup.astype(int)
        )

        combined_score = proba_test * (setup_score_norm + 2 + 0.2 * setup_strength)
        threshold = np.quantile(combined_score, 0.75)

        y_pred = (combined_score >= threshold).astype(int)
        
        #quatrième
        setup_score = (
            0.35 * X_test["trend_up_proba"]
            + 0.25 * X_test["break_up_proba"]
            + 0.15 * X_test["vol_up_proba"]
            + 0.15 * X_test["regime_trend_prob"]
            + 0.10 * X_test["regime_break_prob"]
            - 0.15 * X_test["regime_vol_prob"]
        )

        setup_score_norm = (
            (setup_score - setup_score.mean())
            / (setup_score.std() + 1e-6)
        )
        setup_strength = (
            trend_setup.astype(int)
            + breakout_setup.astype(int)
            + volatility_setup.astype(int)
        )

        combined_score = proba_test * (setup_score_norm + 2 + 0.75 * setup_strength)
        threshold = np.quantile(combined_score, 0.75)

        y_pred = (combined_score >= threshold).astype(int)
       
        

        setup_score = (
            0.35 * X_test["trend_up_proba"]
            + 0.25 * X_test["break_up_proba"]
            + 0.15 * X_test["vol_up_proba"]
            + 0.15 * X_test["regime_trend_prob"]
            + 0.10 * X_test["regime_break_prob"]
            - 0.15 * X_test["regime_vol_prob"]
        )

        setup_score_norm = (
            (setup_score - setup_score.mean())
            / (setup_score.std() + 1e-6)
        )

        setup_strength = (
            trend_setup.astype(int)
            + breakout_setup.astype(int)
            + volatility_setup.astype(int)
        )

        combined_score = proba_test * (setup_score_norm + 2 + 0.75 * setup_strength)

        # seuil absolu + seuil valid
        y_pred = (
            (proba_test >= self.meta_best_threshold)
            & (proba_test >= 0.60)
            & (setup_score_norm > 0)
            & (setup_strength >= 1)
        ).astype(int)
        """
        setup_score = (
            0.35 * X_test["trend_up_proba"]
            + 0.25 * X_test["break_up_proba"]
            + 0.15 * X_test["vol_up_proba"]
            + 0.15 * X_test["regime_trend_prob"]
            + 0.10 * X_test["regime_break_prob"]
            - 0.15 * X_test["regime_vol_prob"]
        )

        setup_score_norm = (
            (setup_score - setup_score.mean())
            / (setup_score.std() + 1e-6)
        )
        combined_score = proba_test * (setup_score_norm + 2)

        threshold = np.quantile(combined_score, 0.75)

        y_pred = (combined_score >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        signal_rate = y_pred.mean()

        print("\n===== META MODEL TEST PERFORMANCE =====")
        print(f"Meta threshold valid : {self.meta_best_threshold:.3f}")
        print("Setup strength counts:")
        #print(setup_strength.value_counts().sort_index())
        print("Combined score stats:")
        print(pd.Series(combined_score).describe())
        print(
            f"Stock name = {self.stock_model.stock_name}\n"
            f"target threshold = {self.thresholds}\n"
            f"horizon = {self.horizon}\n"
        )

        print(f"Proportion de 1 : {y_test.mean():.3f}")
        print(f"Precision       : {precision:.3f}")
        print(f"Recall          : {recall:.3f}")
        print(f"F1-score        : {f1:.3f}")
        print(f"Balanced acc    : {bal_acc:.3f}")
        print(f"Signal rate     : {signal_rate:.3f}")
        print(pd.Series(proba_test).describe())
        print("Nombre de signaux :", y_pred.sum())
        print("Total test :", len(y_pred))
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))

        return y_pred
    
    
    


if __name__ == "__main__":

    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        balanced_accuracy_score,
        accuracy_score
    )

    stocks = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META",
        "GOOGL", "TSLA", "AMD", "QQQ", "SPY"
    ]

    horizons = [10, 20, 30, 45, 60]
    thresholds_list = [0.02, 0.03, 0.05, 0.07]

    start = "2018-01-01"
    end = "2025-01-01"

    all_results = []

    for ticker in stocks:
        for horizon in horizons:
            for thresholds in thresholds_list:

                print("\n" + "=" * 90)
                print(f"TESTING | stock={ticker} | horizon={horizon} | threshold={thresholds}")
                print("=" * 90)

                try:
                    stock_model = Stock_model(ticker, start, end)

                    tendence_model = TrendPriceModel(
                        ticker, start, end, horizon, thresholds
                    )

                    volatile_model = VolatileModel(
                        ticker, start, end, horizon, thresholds
                    )

                    breakout_model = BreakoutModel(
                        ticker, start, end, horizon, thresholds
                    )

                    meta_model = MetaModel(
                        tendence_model,
                        volatile_model,
                        breakout_model,
                        stock_model,
                        thresholds,
                        horizon
                    )

                    meta_model.train_meta_model()
                    y_pred = meta_model.evaluate_meta_model()

                    y_test = meta_model.y_meta_test
                    proba_test = meta_model.model.predict_proba(
                        meta_model.X_meta_test
                    )[:, 1]

                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    accuracy = accuracy_score(y_test, y_pred)

                    all_results.append({
                        "stock": ticker,
                        "horizon": horizon,
                        "target_threshold": thresholds,
                        "best_threshold": meta_model.meta_best_threshold,

                        "test_positive_rate": y_test.mean(),
                        "signal_rate": y_pred.mean(),

                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "balanced_accuracy": bal_acc,
                        "accuracy": accuracy,

                        "proba_mean": proba_test.mean(),
                        "proba_std": proba_test.std(),
                        "proba_min": proba_test.min(),
                        "proba_max": proba_test.max(),

                        "n_test": len(y_test),
                        "n_signals": int(y_pred.sum()),

                        "error": None
                    })

                except Exception as e:
                    print(f"Erreur avec {ticker} | h={horizon} | t={thresholds} : {e}")

                    all_results.append({
                        "stock": ticker,
                        "horizon": horizon,
                        "target_threshold": thresholds,
                        "error": str(e)
                    })

    results_df = pd.DataFrame(all_results)

    all_output_file = "meta_model_all_results_v3.csv"
    results_df.to_csv(all_output_file, index=False)

    clean_df = results_df[results_df["error"].isna()].copy()

    best_2_by_stock = (
        clean_df
        .sort_values(
            by=["stock", "f1", "balanced_accuracy", "precision", "recall"],
            ascending=[True, False, False, False, False]
        )
        .groupby("stock")
        .head(2)
        .reset_index(drop=True)
    )

    best_output_file = "meta_model_best_2_by_stock_v3.csv"
    best_2_by_stock.to_csv(best_output_file, index=False)

    print("\n" + "=" * 90)
    print("MEILLEURES 2 CONFIGURATIONS PAR STOCK")
    print("=" * 90)
    print(best_2_by_stock)

    print(f"\nCSV complet sauvegardé : {all_output_file}")
    print(f"CSV best 2 sauvegardé : {best_output_file}")
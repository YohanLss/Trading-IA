from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
from base_price_model import basePriceModel
from stock_model import Stock_model
from sklearn.metrics import precision_score, recall_score
import warnings
import pandas as pd
import xgboost as xgb
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


class BreakoutModel(basePriceModel):

    def __init__(self, stock, start, end, horizon,thresholds):

        super().__init__(stock,start,end,horizon,thresholds)
        self.stock_model= Stock_model(self.stock,self.start,self.end)
        #self.stock_model.fit()

    
    def get_features(self, include_regime = False):

        df =  super().get_features()


        if include_regime:
            regime_df = self.stock_model.predict_regime_proba("all")
            df = df.join(regime_df)

        if(self.horizon < 10):
            features_breakout_short = [
                "dist_from_high_20","dist_from_low_20","range_width_20","volatility_10",
                "range_expansion_5","return_5d","volume_ratio_5",
                "volume_spike","momentum_volume","price_acceleration"
                ]
            
            for feature in features_breakout_short:
                if(feature in df.columns.tolist() == True ):
                    continue
                else:
                    # distance au plus haut récent
                    df["dist_from_high_20"] = (
                        df["Close"]
                        / df["Close"].rolling(20).max()
                        - 1
                    )

                    # distance au plus bas récent
                    df["dist_from_low_20"] = (
                        df["Close"]
                        / df["Close"].rolling(20).min()
                        - 1
                    )

                    # largeur du range court
                    df["range_width_20"] = (
                        df["Close"].rolling(20).max()
                        - df["Close"].rolling(20).min()
                    ) / df["Close"]

                    # volatilité courte
                    df["volatility_10"] = (
                        df["Close"]
                        .pct_change()
                        .rolling(10)
                        .std()
                    )

                    # expansion du range récent
                    df["range_expansion_5"] = (
                        df["range_width_20"]
                        / (df["range_width_20"].rolling(5).mean() + 1e-6)
                    )

                    # momentum court
                    df["return_5d"] = df["Close"].pct_change(5)

                    # accélération du prix
                    df["price_acceleration"] = (
                        df["return_5d"]
                        - df["return_20d"]
                    )

        elif(self.horizon >= 10 and self.horizon<=60):

            eps = 1e-9

            #features spécifiques breakout
            df["return_5d"] = df["Close"].pct_change(5)
            df["return_10d"] = df["Close"].pct_change(10)
            df["price_acceleration"] = df["return_5d"].diff()

            rolling_high_20 = df["High"].rolling(20).max()
            rolling_low_20 = df["Low"].rolling(20).min()
            rolling_high_50 = df["High"].rolling(50).max()

            df["dist_from_high_20"] = df["Close"] / (rolling_high_20 + eps)
            df["dist_from_high_50"] = df["Close"] / (rolling_high_50 + eps)
            df["range_width_20"] = (rolling_high_20 - rolling_low_20) / (df["Close"] + eps)

            returns = df["Close"].pct_change()
            df["volatility_10"] = returns.rolling(10).std()
            df["volatility_60"] = returns.rolling(60).std()
            df["volatility_compression"] = df["volatility_10"] / (df["volatility_60"] + eps)
            df["atr_expansion"] = df["atr_pct"] / (df["atr_pct"].rolling(20).mean() + eps)

            if "regime_break_prob" in df.columns:
                df["break_regime_strength"] = df["regime_break_prob"] * df["dist_from_high_20"]
                df["break_volume_regime"] = df["regime_break_prob"] * df["volume_spike"]
                df["break_atr_regime"] = df["regime_break_prob"] * df["atr_pct"]
                df["break_compression_regime"] = df["regime_break_prob"] * df["volatility_compression"]

            

            


        else:
            # distance résistance long terme
            df["dist_from_high_100"] = (
                df["Close"]
                / df["Close"].rolling(100).max()
                - 1
            )

            # distance support long terme
            df["dist_from_low_100"] = (
                df["Close"]
                / df["Close"].rolling(100).min()
                - 1
            )

            # volatilité long terme
            df["volatility_60"] = (
                df["Close"]
                .pct_change()
                .rolling(60)
                .std()
            )

            # momentum long terme
            df["return_90d"] = df["Close"].pct_change(90)

            # position relative au trend
            df["dist_sma_100"] = (
                df["Close"]
                / df["sma_100"]
                - 1
            )

            # dispersion long terme
            df["price_dispersion_60"] = (
                df["Close"].rolling(60).std()
                / df["Close"]
            )

            # cohérence du mouvement
            df["trend_consistency"] = (
                df["return_20d"]
                * df["return_60d"]
            )
            


        

        

        

        df = df.drop(columns=["Open", "High", "Low", "Adj Close"], errors="ignore")



        return df
    
    """
    
    def add_target(self):

        df = self.get_features().copy()

        future_return = df["Close"].shift(-self.horizon) / df["Close"] - 1
        

        current_high = df["Close"].rolling(20).max()

        future_close = df["Close"].shift(-self.horizon)

        df["target"] = (
            (future_close > current_high) &
            (future_return > self.thresholds)
        ).astype(int)

        return df.dropna().copy()
    """
    
    def train_model(self):

        super().fit()


        self.model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            random_state=42,
            eval_metric="logloss"
        )


        self.model.fit(self.X_train,self.y_train)

         # ===== FEATURE IMPORTANCE =====
        """"
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.X_train.columns
        ).sort_values(ascending=False)

        print("\n===== FEATURE IMPORTANCE DANS BREAKOUT=====")
        print(importance.head(20))

        leak_cols = [
                    "target",
                    "future_return",
                    "target_trending",
                    "target_volatile",
                    "target_breakout",
                    "target_range"
                ]

        
        print("===== VERIFICATION TARGETS DANS X_train =====")
        for col in leak_cols:
            print(col, "dans X_train ?", col in self.X_train.columns)

        """
        return self
    def leakage_test(self, threshold=0.5):

        from sklearn.utils import shuffle
        from sklearn.metrics import accuracy_score
        import xgboost as xgb

        print("\n===== TEST DATA LEAKAGE =====")

        # copie pour éviter de modifier les vraies données
        X_train = self.X_train.copy()

        # mélange aléatoire de la target
        y_train_shuffled = shuffle(self.y_train, random_state=42)

        # nouveau modèle vierge
        leak_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            random_state=42,
            eval_metric="logloss"
        )

        # entraînement avec target aléatoire
        leak_model.fit(X_train, y_train_shuffled)

        # prédictions sur validation réelle
        proba = leak_model.predict_proba(self.X_valid)[:, 1]
        pred = (proba >= threshold).astype(int)

        acc = accuracy_score(self.y_valid, pred)

        print("Accuracy avec target mélangée :", acc)

        if acc > 0.6:
            print("Possible fuite de données détectée")
        else:
            print(" Pas de fuite évidente")

        return acc
    

    #entraînement sur le k-fold
    def fit_on_data(self,X_train,y_train):

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            random_state=42,
            eval_metric="logloss"
        )

        model.fit(X_train,y_train)

        return model
    
    #prédiction sur le k-fold
    def predict_proba_on_data(self, model, X):

        proba = model.predict_proba(X)[:, 1]

        return proba
    
    def fit_on_data_with_threshold(self, X_train, y_train):

        split = int(0.8 * len(X_train))

        X_inner_train = X_train.iloc[:split]
        y_inner_train = y_train.iloc[:split]

        X_inner_valid = X_train.iloc[split:]
        y_inner_valid = y_train.iloc[split:]

        # modèle pour tuning
        model_inner = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            random_state=42,
            eval_metric="logloss"
        )

        model_inner.fit(X_inner_train, y_inner_train)

        proba_valid = model_inner.predict_proba(X_inner_valid)[:, 1]

        best_threshold = 0.5
        best_score = -1

        for t in np.arange(0.3, 0.9, 0.05):
            pred = (proba_valid >= t).astype(int)

            precision = precision_score(y_inner_valid, pred, zero_division=0)
            recall = recall_score(y_inner_valid, pred, zero_division=0)

            score = precision * recall

            if score > best_score:
                best_score = score
                best_threshold = t

        # modèle final
        model_final = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            random_state=42,
            eval_metric="logloss"
        )

        model_final.fit(X_train, y_train)

        return model_final, best_threshold

    
    def evaluate_valid(self):
        
        self.train_model()
        proba = self.model.predict_proba(self.X_valid)[:,1]
        thresholds = np.arange(0.3,0.8,0.05)

        best_score = 0
        for t in thresholds:

            y_pred = (proba >= t).astype(int)
            precision = precision_score(self.y_valid,y_pred)
            recall = recall_score(self.y_valid,y_pred)
            score = precision * recall

            """"
            print(
                f"threshold {t:.2f}",
                f"precision {precision:.3f}",
                f"recall {recall:.3f}",
                f"score {score:.3f}"
            )
            """

            

            if score > best_score:
                best_score = score
                self.best_threshold = t


        
        #print("\nbest threshold",self.best_threshold)
        


        return self.best_threshold
    
    def evaluate_test(self):
        
        if not hasattr(self, "model"):
            self.train_model()

        if not hasattr(self,"best_threshold"):
            self.evaluate_valid()

        proba = self.model.predict_proba(

            self.X_test

        )[:,1]


        y_pred = (

            proba >= self.best_threshold

        ).astype(int)

        """"

        print(f"Voixi la proportion de 1 :{self.y_test.mean()}")
        print(

            classification_report(

                self.y_test,
                y_pred,
                digits = 3

            )

        )

        """

        return y_pred
    

    def _ensure_ready(self):

        # 1. préparer les datasets si besoin
        if not hasattr(self, "X_train"):
            self.fit()

        # 2. entraîner le modèle si besoin
        if not hasattr(self, "model"):
            self.train_model()

        # 3. calibrer le threshold si besoin
        if not hasattr(self, "best_threshold"):
            self.evaluate_valid()
    

    def evaluate_on_volatile_subset(self, threshold=None, regime_threshold=0.6):

        self.train_model()
        if not hasattr(self, "model"):
            raise ValueError("train_model() doit être appelé avant.")

        if threshold is None:
            self.evaluate_valid()
            threshold = getattr(self, "best_threshold", 0.5)

        

        #target_threshold = getattr(self, "target_threshold", 0.01)
        #print(f" Voici le threshold utilisé sur le trending-subset :{threshold}")
        # récupérer dataset complet avec target
        df = self.add_target().copy()

        # condition : régime volatile dominant
        df["is_breakout"] = (
            (df["regime_break_prob"] > regime_threshold) &
            (df["regime_break_prob"] > df["regime_trend_prob"]) &
            (df["regime_break_prob"] > df["regime_vol_prob"])
        )

        # garder seulement les périodes volatiles
        break_vol = df[df["is_breakout"]].copy()

        #print("\nNombre de lignes breakout :", len(break_vol))

        if len(break_vol) == 0:
            print("Aucune période breakout trouvée")
            return None

        # garder exactement les mêmes colonnes qu'à l'entraînement
        X = break_vol[self.X_train.columns].copy()
        y = break_vol["target"]

        # prédictions
        proba = self.model.predict_proba(X)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        #print("\n===== PERFORMANCE SUR REGIME BREAKOUT =====")
        #print("Proportion de 1 :", y.mean())
        
        from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score

        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        accuracy = accuracy_score(y, y_pred)
        """"
        print("accuracy  :", accuracy)
        print("precision :", precision)
        print("recall    :", recall)

        print("\nclassification report")
        print(classification_report(y, y_pred, digits=3, zero_division=0))
        
        """
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }
    
    def _predict_from_features(self, X, threshold=None):
        self._ensure_ready()

        if not hasattr(self, "model"):
            raise ValueError("Le modèle doit être entraîné avant prédiction.")

        if threshold is None:
            threshold = getattr(self, "best_threshold", 0.5)

        X = X[self.X_train.columns].copy()

        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        return proba, pred


    def predict_history(self, X=None, threshold=None):
        self._ensure_ready()

        if not hasattr(self,"model"):
            raise ValueError("train_model() doit être appelé avant .")

        if X is None:
            X = self.get_features().copy()

        X_model = X[self.X_train.columns].copy()

        proba, pred = self._predict_from_features(X_model, threshold)

        return pd.DataFrame({
            "break_up_proba": proba,
            "break_pred": pred
        }, index=X.index)



    
    def predict_current(self):

        self._ensure_ready()
        if not hasattr(self, "model"):
            raise ValueError("train_model() doit être appelé avant predict_current().")


        df = self.get_features().iloc[[-1]]

        proba = self.model.predict_proba(df)[:,1][0]
        pred = (proba >= self.best_threshold).astype(int)


        return {

            "probability_up_break" : proba,
            "prediction_break" : pred
        }
        

if __name__ == "__main__":

    stock = "QQQ"
    start = "2020-01-01"
    end = "2026-01-25"
    horizon = 15
    thresholds= 0.05

    # création du modèle
    model = BreakoutModel(stock, start, end, horizon, thresholds)

    
    #model.evaluate_test()
    #model.evaluate_on_volatile_subset()
    
    #model.leakage_test()

    print(model.predict_history().head())
    print(model.predict_current())
    




    
    



    


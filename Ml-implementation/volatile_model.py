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

class VolatileModel(basePriceModel):

    def __init__(self, stock, start, end, horizon, thresholds):
        super().__init__(stock, start, end, horizon,thresholds)
        self.stock_model= Stock_model(self.stock,self.start,self.end)
        #self.stock_model.fit()

    def get_features(self,include_regime = False):

        df = super().get_features()

        if include_regime:
            regime_df = self.stock_model.predict_regime_proba("all")
            df = df.join(regime_df)


        #court terme
        if(self.horizon < 10):
            if "regime_prob_vol" in df.columns:
                df["vol_regime_strength"] = df["regime_vol_prob"] * df["volatility_20"]
                df["atr_regime"] = df["regime_vol_prob"] * df["atr_pct"]
                df["volume_vol_regime"] = df["regime_vol_prob"] * df["volume_spike"]

        

        features_vol_long = [

            "volatility_60",

            "volatility_ratio_20_60",

            "volatility_regime_position",

            "price_dispersion_60",

            "return_60d",

            "return_90d",

            "instability_60"

        ]
        #moyen terme
        if(self.horizon>=10 and self.horizon<=60):
            features_vol_medium = ["volatility_10","volatility_20","volatility_ratio_5_20","atr_pct",
                                   "atr_expansion","volatility_compression", "return_20d","momentum_volume",
                                   "volume_ratio_20","volatility_momentum"]
            for feature in features_vol_medium:
                if(feature in df.columns.tolist() == True):
                    continue
                else:

                    df["volatility_5"] =(
                        df["Close"].pct_change().rolling(5).std()
                    )
                    # volatilité moyenne
                    df["volatility_20"] = (
                        df["Close"].pct_change().rolling(20).std()
                    )

                    # ratio court vs moyen terme
                    df["volatility_ratio_5_20"] = (
                        df["volatility_5"]
                        / (df["volatility_20"] + 1e-6)
                    )

                    df["volatility_10"]=(
                        df["Close"].pct_change().rolling(10).std()
                    )

                    df["volatility_50"]=(
                        df["Close"].pct_change().rolling(50).std()
                    )

                    high = df["High"]
                    low = df["Low"]
                    close = df["Close"]
                    hl = high - low
                    hc = (high - close.shift(1)).abs()
                    lc = (low - close.shift(1)).abs()

                    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean()

                    # ATR normalisé
                    df["atr_pct"] = (
                        atr
                        / df["Close"]
                    )

                    # compression de volatilité
                    df["volatility_compression"] = (
                        df["volatility_10"]
                        / (df["volatility_50"] + 1e-6)
                    )

                    # expansion ATR
                    df["atr_expansion"] = (
                        df["atr_pct"]
                        / (df["atr_pct"].rolling(20).mean() + 1e-6)
                    )

                    # momentum volatilité
                    df["volatility_momentum"] = (
                        df["volatility_20"]
                        * df["return_20d"].abs()
                    )

                    df["volume_ratio_20"]=(
                        df["Volume"]
                        / df["Volume"].rolling(20).mean()
                    )

                    # interaction volume volatilité
                    df["volume_volatility"] = (
                        df["volume_ratio_20"]
                        * df["volatility_20"]
                    )
            
            else:
                for feature in features_vol_long:
                    if(feature in df.columns.tolist() == True):
                        continue
                    else:
                        # volatilité long terme
                        df["volatility_60"] = (
                            df["Close"].pct_change().rolling(60).std()
                        )

                        # ratio moyen vs long terme
                        df["volatility_ratio_20_60"] = (
                            df["volatility_20"]
                            / (df["volatility_60"] + 1e-6)
                        )

                        volatility_20 = (
                            df["Close"].pct_change().rolling(20).std()
                        )

                        # position de la volatilité actuelle
                        df["volatility_regime_position"] = (
                            volatility_20
                            / (volatility_20.rolling(100).mean() + 1e-6)
                        )

                        # dispersion du prix
                        df["price_dispersion_60"] = (
                            df["Close"].rolling(60).std()
                            / df["Close"]
                        )

                        
                

            



       
        #df["accel_vol_regime"] = df["regime_vol_prob"] * df["price_acceleration"]

        

        cols_to_drop = ["High", "Low", "Open","Adj Close","Volume"]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        

        return df
        

    """"
    def add_target(self):

        df = self.get_features().copy()

        future_return = df["Close"].shift(-self.horizon) / df["Close"] - 1
        

        df["target"] = (
            future_return.abs() > self.thresholds
        ).astype(int)

        return df.dropna().copy()

    """

    def time_split(self):
        df= super().add_target()
        df = df.drop(columns=["Close"], errors="ignore")
        n=len(df)
        print(df.columns.tolist())

        train = df.iloc[:int(0.6*n)]
        valid = df.iloc[int(0.6*n):int(0.8*n)]
        test  = df.iloc[int(0.8*n):]


        return train,valid,test
    


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

        print("\n===== FEATURE IMPORTANCE DANS VOLATILITY =====")
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
            print("Pas de fuite évidente")

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

            """
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
        proba = self.model.predict_proba(self.X_test)[:,1]


        y_pred = (proba >= self.best_threshold).astype(int)
        """
        print(f"Stock name :{self.stock} \n")
        print(f"périodes de {self.start} à {self.end}")
        print(f"Voici la proportion de 1 :{self.y_test.mean()}")
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
        print(f" Voici le threshold utilisé sur le volatile-subset :{threshold}")

        

        # récupérer dataset complet avec target
        df = self.add_target().copy()

        # condition : régime volatile dominant
        df["is_volatile"] = (
            (df["regime_vol_prob"] > regime_threshold) &
            (df["regime_vol_prob"] > df["regime_trend_prob"]) &
            (df["regime_vol_prob"] > df["regime_break_prob"])
        )

        # garder seulement les périodes volatiles
        vol_df = df[df["is_volatile"]].copy()

        print("\nNombre de lignes volatile :", len(vol_df))

        if len(vol_df) == 0:
            print("Aucune période volatile trouvée")
            return None

        # garder exactement les mêmes colonnes qu'à l'entraînement
        X = vol_df[self.X_train.columns].copy()
        y = vol_df["target"]

        # prédictions
        proba = self.model.predict_proba(X)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        print("\n===== PERFORMANCE SUR REGIME VOLATILE =====")
        print("Proportion de 1 :", y.mean())

        from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score

        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        accuracy = accuracy_score(y, y_pred)

        print("accuracy  :", accuracy)
        print("precision :", precision)
        print("recall    :", recall)

        print("\nclassification report")
        print(classification_report(y, y_pred, digits=3, zero_division=0))

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
            "vol_up_proba": proba,
            "vol_pred": pred
        }, index=X.index)



    
    def predict_current(self):

        self._ensure_ready()
        if not hasattr(self, "model"):
            raise ValueError("train_model() doit être appelé avant predict_current().")

        df = self.get_features().iloc[[-1]]
        
        # garder uniquement la dernière ligne
        df_last = df.iloc[[-1]]

        # garder exactement les mêmes colonnes qu'au train
        X = df_last[self.X_train.columns]

        """

        print("Colonnes train :", self.X_train.columns.tolist())
        print("Nb train :", len(self.X_train.columns))

        print("Colonnes current :", X.columns.tolist())
        print("Nb current :", len(X.columns))

        extra = [c for c in df.columns if c not in self.X_train.columns]
        missing = [c for c in self.X_train.columns if c not in df.columns]

        print("Colonnes en trop :", extra)
        print("Colonnes manquantes :", missing)

        """

        proba = self.model.predict_proba(X)[:,1][0]
        pred = (proba >= self.best_threshold).astype(int)


        return {

            "probability_up_vol" : proba,
            "prediction_vol" : pred
        }
        

if __name__ == "__main__":

    stock = "TSLA"
    start = "2020-01-01"
    end = "2026-01-25"
    horizon = 5
    thresholds= 0.05

    # création du modèle
    model = VolatileModel(stock, start, end, horizon,thresholds)

    #model.train_model()
    #model.evaluate_valid()
    model.evaluate_test()
    model.evaluate_on_volatile_subset()
    print(model.predict_history().head())
    print(model.predict_current())
    model.evaluate_test()
    #print(model.predict_history().head())
    
    #model.leakage_test()

    #print(model.predict_history().head())
    

    


    
    #model.evaluate_on_volatile_subset(threshold=None,regime_threshold=0.6)


    
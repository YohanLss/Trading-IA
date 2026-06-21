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



class TrendPriceModel(basePriceModel):


    def __init__(self, stock, start, end, horizon,thresholds):

        super().__init__(stock,start,end,horizon,thresholds)
        self.stock_model= Stock_model(self.stock,self.start,self.end)
        #self.stock_model.fit()
        

        


    
    
    def get_features(self, include_regime =False):

        df = super().get_features()

        if include_regime:
            regime_df = self.stock_model.predict_regime_proba("all")
            df = df.join(regime_df)
        

        #ajout des features court terme sur trending datatset
        if(self.horizon < 10):
            columns_short_trend=["return_5d","price_acceleration","momrntum_volume",
                             "sma_20_slope","voulume_spike"]
            

            
            for column in columns_short_trend:
                if(column in df.columns.tolist() == True):
                    continue
                else:

                    sma_20=df["Close"].rolling(20).mean()
                    # momentum court terme
                    df["return_5d"] = df["Close"].pct_change(5)
                    df["return_20d"] = df["Close"].pct_change(20)

                    # accélération du prix
                    df["price_acceleration"] = (
                        df["return_5d"]
                        - df["return_20d"]
                    )

                    # momentum rapide vs lent
                    df["momentum_ratio_5_20"] = (df["return_5d"]/ (df["return_20d"] + 1e-6))

                    # pente rapide
                    df["sma_20_slope"] = (sma_20.pct_change(5))

                    volume_ratio_5 = (
                        df["Volume"]
                        / df["Volume"].rolling(5).mean()
                    )

                    # confirmation volume court terme
                    df["momentum_volume_short"] = (volume_ratio_5 * df["return_5d"])
        

        elif(self.horizon >=10 and self.horizon<=60):
            # features spécifiques trend
        
            df["trend_momentum"] = df["sma_50_slope"] * df["return_20d"]
            

            
            
            df["trend_confirmation"] = df["golden_cross"] * df["trend_strength"]
            


            if "regime_trend_prob" in df.columns:
                df["trend_regime_strength"] = (
                    df["trend_strength"] * df["regime_trend_prob"]
                )

                df["trend_quality"] = df["trend_strength"] * df["momentum_ratio_20_60"]

                df["trend_slope_regime"] = (
                    df["regime_trend_prob"] * df["sma_50_slope"]
                )

                df["momentum_regime"] = (
                    df["regime_trend_prob"] * df["momentum_ratio_20_60"]
                )

                df["stability_regime"] = (
                    df["regime_trend_prob"] * df["trend_stability"]
                )

        else:

            df["return_20d"] = df["Close"].pct_change(20)
            sma_100=df["Close"].rolling(100).mean()
            # distance à la moyenne longue
            df["dist_sma_100"] = (
                df["Close"]
                / sma_100
                - 1
            )

            # position dans la tendance long terme
            df["dist_from_high_100"] = (
                df["Close"]
                / df["Close"].rolling(100).max()
                - 1
            )

            # momentum long terme
            df["return_90d"] = df["Close"].pct_change(90)

            # ratio momentum long vs moyen
            df["momentum_ratio_20_90"] = (
                df["return_20d"]
                / (df["return_90d"] + 1e-6)
            )

            # cohérence du trend
            df["trend_consistency"] = (
                df["return_20d"]
                * df["return_60d"]
            )
        
        

        cols_to_drop = ["High", "Low", "Open","Adj Close"]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        return df
    """
    def add_target(self):

        df = self.get_features().copy()

        future_return = df["Close"].shift(-self.horizon) / df["Close"] - 1
        

        df["target"] = (
            (future_return > self.thresholds) &
            (df["trend_strength"] > df["trend_strength"].rolling(50).median()) &
            (df["sma_50_slope"] > 0)
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

        print("\n===== FEATURE IMPORTANCE DANS TENDENCE =====")
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

        
        """
        y_pred = (proba_valid >= best_threshold).astype(int)
        
        print("--------- DEBUG INNER VALID tendence ----------\n")
        print(f"Stock name : {self.stock}\n")
        print(f"Période de {self.start} à {self.end}")
        print(f"Best threshold : {best_threshold:.2f}")
        print(f"Signal rate valid : {y_pred.mean():.3f}")

        print(
            classification_report(
                y_inner_valid,
                y_pred,
                digits=3,
                zero_division=0
            )
        )
        """

        model_final.fit(X_train, y_train)

        return model_final, best_threshold


    def evaluate_valid(self):
        
        self.train_model()
        proba = self.model.predict_proba(self.X_valid)[:,1]
        thresholds = np.arange(0.3,0.8,0.05)
        best_score = -1


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


        
        print(

            "\nbest threshold",
            self.best_threshold

        )
        


        return self.best_threshold


    def evaluate_test(self):


        if not hasattr(self, "model"):
            self.train_model()

        if not hasattr(self,"best_threshold"):
            self.evaluate_valid()
        proba = self.model.predict_proba(self.X_test)[:,1]


        y_pred = (proba >= self.best_threshold).astype(int)
        """"

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
    
    def evaluate_on_trending_subset(self, threshold=None, regime_threshold=0.6):

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


        # condition : regime trending dominant
        df["is_trending"] = (
            (df["regime_trend_prob"] > regime_threshold) &
            (df["regime_trend_prob"] > df["regime_vol_prob"]) &
            (df["regime_trend_prob"] > df["regime_break_prob"])
        )

        # garder seulement périodes trending
        trend_df = df[df["is_trending"]].copy()

        print("\nNombre de lignes trending :", len(trend_df))

        if len(trend_df) == 0:
            print("Aucune période trending trouvée")
            return None

        # séparer X et y
        X = trend_df.drop(columns=["target", "is_trending"], errors="ignore")
        y = trend_df["target"]

        # prédictions
        proba = self.model.predict_proba(X)[:,1]

        y_pred = (proba >= threshold).astype(int)

        print("\n===== PERFORMANCE SUR REGIME TRENDING =====")
        print("Proportion de 1 :", y.mean())

        from sklearn.metrics import classification_report, precision_score, recall_score

        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        """

        print(f"Stock name :{self.stock} \n")
        print(f"périodes de {self.start} à {self.end}")
        print("precision :", precision)
        print("recall :", recall)

        print("\nclassification report")
        print(classification_report(y, y_pred, digits=3))
        """

        return precision, recall
        
       

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
        
        

        #la prédiction finale se fait avec les mêmes colonnes de training set
        X_model = X[self.X_train.columns].copy()

        proba, pred = self._predict_from_features(X_model, threshold)

        

        return pd.DataFrame({
            "trend_up_proba": proba,
            "trend_pred": pred
        }, index=X.index)


    
    def predict_current(self):

        self._ensure_ready()
        if not hasattr(self, "model"):
            raise ValueError("train_model() doit être appelé avant predict_current().")

        df = self.get_features().iloc[[-1]]

        X_model = df[self.X_train.columns]

        proba = self.model.predict_proba(X_model)[:, 1][0]

        threshold = getattr(self, "best_threshold", 0.5)

        pred = int(proba >= threshold)

        return {
            "probability_up": proba,
            "prediction": pred
        }
    

if __name__ == "__main__":

    stock = "AAPL"
    start = "2020-01-01"
    end = "2026-01-25"
    horizon = 5
    thresholds = 0.05

    # création du modèle
    model = TrendPriceModel(stock, start, end, horizon ,thresholds)


    

    model.evaluate_on_trending_subset()
    model.evaluate_test()
    print(model.predict_history())
    print(model.predict_current())
    #model.leakage_test()
    
    """"
    print("\nFeatures dans leur totalité, avec la target et le nettoyage :")
    print(df.head())
    print(df.shape)
    
    # ajout des features de régime
    #df = model.add_regime_features(df)

    #print("\nFeatures après ajout du régime :")
    #print(df.head())

    
    print("\nColonnes disponibles :")
    print(df.columns.tolist())

    print("\nStatistiques des nouvelles features :")
    print(df[[
        "regime_trend_prob",
        "regime_vol_prob",
        "regime_break_prob"
    ]].describe())

    print("\nNombre de NaN par colonne :")
    print(df[[
        "regime_trend_prob",
        "regime_vol_prob",
        "regime_break_prob"
    ]].isna().sum())

    print("\nDimensions finales du dataset :")
    print(df.shape)

    """


    
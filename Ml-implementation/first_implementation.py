import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, recall_score


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# 1) Télécharger les données
def load_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    df = df.sort_index()

    # MultiIndex: (Price, Ticker) -> on garde Price uniquement si 1 ticker
    if isinstance(df.columns, pd.MultiIndex):
        # si 1 ticker
        df = df.xs(ticker, axis=1, level=1)

    return df



# 2) Feature engineering
def add_features(df, horizon):
    df = df.copy()
    eps = 1e-9

    close = df["Close"]
    volume = df["Volume"]
    ret = close.pct_change(fill_method=None)

    # ===== LONG TERME =====
    if horizon >= 90:
        df["return_90d"]   = close.pct_change(90,  fill_method=None)
        df["return_180d"]  = close.pct_change(180, fill_method=None)
        df["return_252d"]  = close.pct_change(252, fill_method=None)

        df["sma_200"] = close.rolling(200).mean()
        df["dist_sma_200"] = (close - df["sma_200"]) / (df["sma_200"] + eps)
        df["sma_200_slope"] = df["sma_200"].pct_change(20, fill_method=None)
        df["above_sma_200"] = (close > df["sma_200"]).astype(int)

        df["momentum_mean_90"] = ret.rolling(90).mean()
        df["volatility_90"]    = ret.rolling(90).std()
        df["volatility_180"]   = ret.rolling(180).std()

        delta = close.diff()
        gain60 = delta.clip(lower=0).rolling(60).mean()
        loss60 = (-delta.clip(upper=0)).rolling(60).mean()
        df["RSI_60"] = 100 - (100 / (1 + gain60 / (loss60 + eps)))

        gain90 = delta.clip(lower=0).rolling(90).mean()
        loss90 = (-delta.clip(upper=0)).rolling(90).mean()
        df["RSI_90"] = 100 - (100 / (1 + gain90 / (loss90 + eps)))

        df["drawdown_252"] = close / close.rolling(252).max() - 1

        df["bull_market"] = (
            (close > df["sma_200"]) &
            (df["sma_200_slope"] > 0)
        ).astype(int)

        df["sma_50"] = close.rolling(50).mean()
        df["sma_100"] = close.rolling(100).mean()
        df["dist_sma_50"] = (close - df["sma_50"]) / (df["sma_50"] + eps)
        df["dist_sma_100"] = (close - df["sma_100"]) / (df["sma_100"] + eps)
        df["ratio_52w_high"] = close / (close.rolling(252).max() + eps)


        df["ann_vol_60"] = ret.rolling(60).std() * np.sqrt(252)
        df["trend_strength_50_200"] = df["sma_50"] / (df["sma_200"] + eps)


    

    # ===== MOYEN TERME =====
    elif horizon >= 10:
        df["return_10d"] = close.pct_change(10, fill_method=None)
        df["return_20d"] = close.pct_change(20, fill_method=None)

        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_60 = close.rolling(60).mean()

        ema_10 = close.ewm(span=10, adjust=False).mean()
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_60 = close.ewm(span=60, adjust=False).mean()

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + eps)
        df["RSI"] = 100 - (100 / (1 + rs))

        df["volatility_20"] = ret.rolling(20).std()

        # Volume MAs
        df["vma_5"]  = volume.rolling(5).mean()
        df["vma_10"] = volume.rolling(10).mean()
        df["vma_20"] = volume.rolling(20).mean()
        df["vma_60"] = volume.rolling(60).mean()

        df["price_ratio_10_20"] = sma_10 / (sma_20 + eps)
        df["price_ratio_20_60"] = sma_20 / (sma_60 + eps)

        df["vol_ratio_10_20"] = df["vma_10"] / (df["vma_20"] + eps)
        df["vol_ratio_20_60"] = df["vma_20"] / (df["vma_60"] + eps)
        df["vol_ratio_5_20"]  = df["vma_5"]  / (df["vma_20"] + eps)
        df["vol_ratio_5_60"]  = df["vma_5"]  / (df["vma_60"] + eps)

        df["ema_ratio_10_20"] = ema_10 / (ema_20 + eps)
        df["ema_ratio_20_60"] = ema_20 / (ema_60 + eps)

        df["momentum_accel_20"] = df["return_20d"].diff()
        df["breakout_20"] = close / close.rolling(20).max()
        df["breakout_60"] = close / close.rolling(60).max()

        df["rsi_velocity"] = df["RSI"].diff()

        df["ema_20_50"]  = close.ewm(span=20).mean() - close.ewm(span=50).mean()
        df["ema_50_100"] = close.ewm(span=50).mean() - close.ewm(span=100).mean()

        

    # ===== COURT TERME =====
    else:
        df["return_5d"] = close.pct_change(5, fill_method=None)
        df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + eps)
        v = df["Volume"]
        df["vol_z_20"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + eps)


        prev_close = df["Close"].shift(1)
        df["gap"] = (df["Open"] - prev_close) / (prev_close + eps)


        sma_5  = close.rolling(5).mean()
        sma_10 = close.rolling(10).mean()

        ema_5  = close.ewm(span=5, adjust=False).mean()
        ema_10 = close.ewm(span=10, adjust=False).mean()

        prev_close = df["Close"].shift(1)
        df["gap"] = (df["Open"] - prev_close) / (prev_close + eps)


        df["vma_5"]  = volume.rolling(5).mean()
        df["vma_10"] = volume.rolling(10).mean()

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + eps)
        df["RSI"] = 100 - (100 / (1 + rs))

        df["ema_ratio_5_10"] = ema_5 / (ema_10 + eps)
        df["price_ratio_5_10"] = sma_5 / (sma_10 + eps)
        df["vol_ratio_5_10"] = df["vma_5"] / (df["vma_10"] + eps)
        df["momentum_accel_5"] = df["return_5d"].diff()

        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        df["stoch_k_14"] = (df["Close"] - low14) / (high14 - low14 + eps)


    return df



# 3) Target (return futur + seuil)
def add_target(df, horizon=60, thresholds=0.05):
    df = df.copy()
    df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target"] =  (df["future_return"] >= thresholds).astype(int)
    return df


# 4) Nettoyage : supprimer fuite + NaN
def clean_dataset(df):
    df = df.copy()

    #évite les fuites
    df = df.dropna()
    df = df.drop(columns=["future_return"], errors="ignore")  
    
    return df


# 5) Split chronologique
def time_split(df):
    n = len(df)

    train = df.iloc[:int(0.6*n)]
    valid = df.iloc[int(0.6*n):int(0.8*n)]
    test  = df.iloc[int(0.8*n):]

    X_train = train.drop(columns=["target"])
    y_train = train["target"]

    X_valid = valid.drop(columns=["target"])
    y_valid = valid["target"]

    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# 6) Standardisation (fit sur train seulement)
def scale_sets(X_train, X_valid, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler


# 7) Entraîner + évaluer RandomForest
def train_eval_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": model.score(X_test, y_test),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, digits=3)
    }
    return model, y_pred, metrics

#ensemble des modèles à utiliser
def get_model(model_name="rf",scale_pos_weight=0):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

    elif model_name == "logreg":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        )

    elif model_name == "xgb":
        return xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic" ,       #"multi:softprob",
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    #peut-être faire plusieurs tests pour knn
    elif model_name == "knn":
        return KNeighborsClassifier(n_neighbors=7 ,
                                     weights="distance")

    else:
        raise ValueError(f"Unknown model: {model_name}")


def eval_thresholds_xgb(model, X_test, y_test, thresholds):
    proba = model.predict_proba(X_test)[:, 1]
    rows = []

    for t in thresholds:
        y_pred = (proba >= t).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        score_A = precision * recall
        score_B = precision * recall * bal_acc

        rows.append({
            "decision_threshold": t,
            "precision": precision,
            "recall": recall,
            "balanced_accuracy": bal_acc,
            "score_A": score_A,
            "score_B": score_B
        })

    return pd.DataFrame(rows)



def train_eval_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, model_name):
    y_train = np.asarray(y_train).astype(int)
    y_valid = np.asarray(y_valid).astype(int)
    y_test  = np.asarray(y_test).astype(int)

    if model_name == "xgb":
        model.fit(X_train, y_train)

        thresholds_to_test = [0.1,0.15,0.2, 0.25, 0.3, 0.35, 0.4,0.45,0.5]

        # 1) choisir threshold sur VALID
        df_thr = eval_thresholds_xgb(model, X_valid, y_valid, thresholds_to_test)
        best = df_thr.sort_values("score_B", ascending=False).iloc[0]
        best_threshold = float(best["decision_threshold"])

        # 2) eval finale sur TEST avec ce threshold
        proba_test = model.predict_proba(X_test)[:, 1]
        y_pred = (proba_test >= best_threshold).astype(int)

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        best_threshold = np.nan

    return {
        "accuracy": (y_pred == y_test).mean(),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "report": classification_report(y_test, y_pred, labels=[0, 1], digits=3, zero_division=0),
        "best_decision_threshold": best_threshold
    }



# 8) Boucle thresholds (et optionnellement horizons)
def test_thresholds(
    ticker="AAPL",
    start="2018-01-01",
    end="2026-01-01",
    horizon=(1,5,10),
    thresholds=(0.02, 0.03, 0.05, 0.07),
    verbose=True,
    model_name ="rf"
):
    raw = load_data(ticker, start, end)
    
   

    results = []

    for h in horizon:
        df_feat = raw.copy()
        df_feat = add_features(df_feat,horizon=h)
        for th in thresholds:
            df = df_feat.copy()
            df = add_target(df, horizon=h, thresholds=th)
            df = clean_dataset(df)

            # si dataset trop petit ou target constant, on skip
            
            if df["target"].nunique() < 2 or len(df) < 200:
                results.append({
                    "threshold": th,
                    "n_samples": len(df),
                    "target_rate": df["target"].mean() if len(df) else np.nan,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "note": "skip (target constant ou trop peu de données)",
                    "precision": np.nan,
                    "recall": np.nan

                })
                continue

            


            X_train, y_train, X_valid, y_valid, X_test, y_test = time_split(df)
           
            X_train_s, X_valid_s, X_test_s, scaler = scale_sets(X_train, X_valid, X_test)

            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            scale_pos_weight = (neg / pos) if pos > 0 else 1

            model = get_model(model_name, scale_pos_weight=scale_pos_weight)

            
            
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                results.append({
                    "stock": ticker,
                    "model": model_name,
                    "horizon": h,
                    "threshold": th,
                    "n_samples": len(df),
                    "target_rate": df["target"].mean(),
                    "precision": np.nan,
                    "recall": np.nan,
                    "note": "skip (une seule classe dans train ou test)"
                })
                continue

            

            
            

            needs_scaling = model_name in ["logreg", "knn"] 

            #scaling utile que pour knn et logreg

            if needs_scaling:
                metrics = train_eval_model(
                model,
                X_train_s, y_train,
                X_valid_s, y_valid,
                X_test_s, y_test,
                model_name=model_name
)

            else:
                metrics = train_eval_model(
                model,
                X_train, y_train,
                X_valid, y_valid,
                X_test, y_test,
                model_name=model_name
            )



            row = {
                "stock": ticker,
                "model": model_name,
                "horizon": h,
                "threshold": th,
                "n_samples": len(df),
                "target_rate": df["target"].mean(),
                "recall":metrics["recall"],#"accuracy": metrics["accuracy"],
                "precision": metrics["precision"] ,#"balanced_accuracy": metrics["balanced_accuracy"],
                "decision_threshold": metrics.get("best_decision_threshold", np.nan),
                "score_B": metrics["precision"] * metrics["recall"] * metrics["balanced_accuracy"]

            }
            results.append(row)
            
            rate_pos = (df["target"] == 1).mean()
            rate_neg = (df["target"] == 0).mean()
            #rate_neu = (df["target"] == 0).mean()


            if verbose:
                print("=" * 60)
                print(f"ticker={ticker} | model={model_name} | threshold={th} | horizon={h}")
                print(f"n={row['n_samples']} | "f"pos={rate_pos:.3f} | "f"neg={rate_neg:.3f}")    #| "f"neu={rate_neu:.3f} 
                print(f"precision={row['precision']:.3f} | recall={row['recall']:.3f}")
                print(metrics["confusion_matrix"])
                print(metrics["report"])

            
    return pd.DataFrame(results).sort_values(by="score_B", ascending=False)


models = ["logreg", "knn", "rf","xgb"]

stocks = ["AAPL","MSFT","NVDA","AMZN","TSLA","META","TSM"]

for s in stocks:
    all_results = []


    print("------" + s + "-------")
    for m in models:

        df_res = test_thresholds(ticker=s,start="2020-01-01",
            end="2026-01-01",model_name=m, verbose=True)
        all_results.append(df_res)

    final = pd.concat(all_results).sort_values("score_B", ascending=False)
    print(final.head(20))
    final.to_csv( "simulation_results/" +s+"_results"+".csv", index = False)





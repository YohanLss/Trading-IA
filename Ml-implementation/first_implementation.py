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
def add_features(df):
    df = df.copy()

    # returns
    
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # moving averages (SMA)
    sma_5 = df["Close"].rolling(5).mean()
    sma_10 = df["Close"].rolling(10).mean()
    sma_20 = df["Close"].rolling(20).mean()
    sma_30 = df["Close"].rolling(30).mean()
    sma_60 = df["Close"].rolling(60).mean()
   

    # EMA
    ema_5 = df["Close"].ewm(span=5, adjust=False).mean()
    ema_10 = df["Close"].ewm(span=10, adjust=False).mean()
    ema_20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema_30 = df["Close"].ewm(span=30, adjust=False).mean()
    ema_60 = df["Close"].ewm(span=60, adjust=False).mean()


    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # moyenne glissante sur 14 jours
    window = 14
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    #division par zéro
    rs = avg_gain / avg_loss


    # RSI
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSA"] = df["Close"].pct_change(12)

    # Volatilité
    df["volatility_10"] = df["return_5d"].rolling(50).std()

    #ATR
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = abs(df["High"] - df["Close"].shift(1))
    df["L-C"] = abs(df["Low"] - df["Close"].shift(1))
    df["tr"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()


    #Volume indicators
    df["volume_ma_10"] = df["Volume"].rolling(10).mean()

    #features de Ben-----Début
    df["price_ratio_5_10"] = sma_5 / sma_10
    df["price_ratio_10_20"] = sma_10 / sma_20
    df["price_ratio_20_60"] = sma_20 / sma_60

    df["vma_5"] = df["Volume"].rolling(5).mean()
    df["vma_10"] = df["Volume"].rolling(10).mean()
    df["vma_20"] = df["Volume"].rolling(20).mean()
    df["vma_60"] = df["Volume"].rolling(60).mean()

    df["vol_ratio_5_10"] = df["vma_5"] / df["vma_10"]
    df["vol_ratio_10_20"] = df["vma_10"] / df["vma_20"]
    df["vol_ratio_20_60"] = df["vma_20"] / df["vma_60"]
    df["vol_ratio_5_20"] = df["vma_5"] / df["vma_20"]
    df["vol_ratio_5_60"] = df["vma_5"] / df["vma_60"]

    df["ema_ratio_5_10"] = ema_5 / ema_10
    df["ema_ratio_10_20"] = ema_10 / ema_20
    df["ema_ratio_5_20"] = ema_5 / ema_20
    df["ema_ratio_20_60"] = ema_20 / ema_60


    

    #features de Ben-----Fin
    # Long-term trend
    df["sma_200"] = df["Close"].rolling(200).mean()

    # Distance au régime long terme
    df["dist_sma_200"] = (df["Close"] - df["sma_200"]) / df["sma_200"]

    # Pente de la SMA200 (direction du marché)
    df["sma_200_slope"] = df["sma_200"].pct_change(20)

    # Position relative
    df["above_sma_200"] = (df["Close"] > df["sma_200"]).astype(int)

    # Multi-horizon returns
    df["return_60d"] = df["Close"].pct_change(60)

    # RSI long terme
    df["RSI_30"] = 100 - (100 / (1 + (
    df["Close"].diff().clip(lower=0).rolling(30).mean() /
        (-df["Close"].diff().clip(upper=0).rolling(30).mean())
        )))

    # Trend alignment
    df["ema_20_50"] = df["Close"].ewm(span=20).mean() - df["Close"].ewm(span=50).mean()
    df["ema_50_100"] = df["Close"].ewm(span=50).mean() - df["Close"].ewm(span=100).mean()


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

"""""
def encode_y_for_xgb(y):
    # -1 -> 0, 0 -> 1, 1 -> 2
    return (y + 1).astype(int)

def decode_y_from_xgb(y_pred):
    # 0 -> -1, 1 -> 0, 2 -> 1
    return (y_pred - 1).astype(int)
"""

#ensemble des modèles à utiliser
def get_model(model_name="rf"):
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
        #num_class=2,
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




def train_eval_model(model, X_train, y_train, X_test, y_test, model_name):
    y_train = np.asarray(y_train).astype(int)
    y_test  = np.asarray(y_test).astype(int)

    assert len(X_train) == len(y_train), f"Train mismatch: {len(X_train)} vs {len(y_train)}"
    assert len(X_test) == len(y_test), f"Test mismatch: {len(X_test)} vs {len(y_test)}"

    if model_name == "xgb":
        model.fit(X_train, y_train)  # y_train doit être 0/1
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    return {
        "accuracy": (y_pred == y_test).mean(),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "report": classification_report(y_test, y_pred, labels=[0, 1], digits=3, zero_division=0)
    }


# 8) Boucle thresholds (et optionnellement horizons)
def test_thresholds(
    ticker="AAPL",
    start="2020-01-01",
    end="2026-01-01",
    horizon=(1,3,5,10,20, 45, 60, 90),
    thresholds=(0.01, 0.02, 0.03, 0.05, 0.07 , 0.10,0.15,0.18,0.20),
    verbose=True,
    model_name ="rf"
):
    raw = load_data(ticker, start, end)
    raw = add_features(raw)
   

    results = []

    for h in horizon:
        for th in thresholds:
            df = add_target(raw, horizon=h, thresholds=th)
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

            
            model = get_model(model_name)
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
                metrics = train_eval_model(model, X_train_s, y_train, X_test_s, y_test, model_name=model_name)
            else:
                metrics = train_eval_model(model, X_train, y_train, X_test, y_test, model_name=model_name)


            row = {
                "stock": ticker,
                "model": model_name,
                "horizon": h,
                "threshold": th,
                "n_samples": len(df),
                "target_rate": df["target"].mean(),
                "recall":metrics["recall"],#"accuracy": metrics["accuracy"],
                "precision": metrics["precision"] #"balanced_accuracy": metrics["balanced_accuracy"],
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

            
    return pd.DataFrame(results).sort_values(by="precision", ascending=False)


models = ["logreg", "knn", "rf","xgb"]
all_results = []
stocks = ["AAPL","MSTF","SPY","QQQ","TSLA","META"]

for s in stocks:
    print("------" + s + "-------")
    for m in models:
        df_res = test_thresholds(ticker=s,start="2020-01-01",
            end="2026-01-01",model_name=m, verbose=True)
        all_results.append(df_res)

    final = pd.concat(all_results).sort_values("recall", ascending=False)
    print(final.head(20))
    final.to_csv( "simulation_results/" +s+"_results"+".csv", index = False)





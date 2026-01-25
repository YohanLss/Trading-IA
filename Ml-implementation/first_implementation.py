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


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# 1) Télécharger les données
def load_data(ticker="AAPL", start="2020-01-01", end="2024-01-01"):
    donne = yf.download(ticker, start=start, end=end, auto_adjust = True)
    donne = donne.sort_index()
    return donne


# 2) Feature engineering
def add_features(df):
    df = df.copy()

    # returns
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # moving averages (SMA)
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_30"] = df["Close"].rolling(30).mean()

    # EMA
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema_30"] = df["Close"].ewm(span=30, adjust=False).mean()

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
    df["volatility_10"] = df["return_1d"].rolling(10).std()

    #ATR
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = abs(df["High"] - df["Close"].shift(1))
    df["L-C"] = abs(df["Low"] - df["Close"].shift(1))
    df["tr"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()


    #Volume indicators
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma_10"] = df["Volume"].rolling(10).mean()

    return df


# 3) Target (return futur + seuil)
def add_target(df, horizon=60, thresholds=0.05):
    df = df.copy()
    df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target"] =  np.select([df["future_return"] >= thresholds, df["future_return"] <= -thresholds],
    [1, -1],
    default=0
).astype(int)
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


def encode_y_for_xgb(y):
    # -1 -> 0, 0 -> 1, 1 -> 2
    return (y + 1).astype(int)

def decode_y_from_xgb(y_pred):
    # 0 -> -1, 1 -> 0, 2 -> 1
    return (y_pred - 1).astype(int)

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
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    #peut-être faire plusieurs tests pour knn
    elif model_name == "knn":
        return KNeighborsClassifier(n_neighbors=7 ,
                                     weights="distance")

    else:
        raise ValueError(f"Unknown model: {model_name}")



def train_eval_model(model, X_train, y_train, X_test, y_test, model_name, labels=[-1,0,1]):
    if model_name == "xgb":
        y_train_fit = (y_train + 1).astype(int)
        model.fit(X_train, y_train_fit)
        y_pred = (model.predict(X_test).astype(int) - 1).astype(int)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    bal_acc = recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)

    return {
        "model": model,
        "y_pred": y_pred,
        "accuracy": (y_pred == y_test).mean(),
        "balanced_accuracy": bal_acc,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "report": classification_report(y_test, y_pred, labels=labels, digits=3, zero_division=0)
    }



# 8) Boucle thresholds (et optionnellement horizons)
def test_thresholds(
    ticker="AAPL",
    start="2020-01-01",
    end="2024-01-01",
    horizon=(20, 45, 60, 90),
    thresholds=(0.03, 0.05, 0.07, 0.10),
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
                    "note": "skip (target constant ou trop peu de données)"
                })
                continue

            X_train, y_train, X_valid, y_valid, X_test, y_test = time_split(df)

            # skip si le test ne contient pas toutes les classes
            if y_test.nunique() < 3:
                results.append({
                    "model": model_name,
                    "horizon": h,
                    "threshold": th,
                    "n_samples": len(df),
                    "target_rate": df["target"].mean(),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "note": "skip (classes manquantes dans y_test)"
                })
                continue

            X_train_s, X_valid_s, X_test_s, scaler = scale_sets(X_train, X_valid, X_test)

            # RandomForest ne nécessite pas scaling, mais ce n’est pas grave.
            # Si tu veux "propre", on peut donner les X_* non-scalés au RF.
            model = get_model(model_name)
        

            needs_scaling = model_name in ["logreg", "knn"] 

            #scaling utile que pour knn et logreg

            if needs_scaling:
                metrics = train_eval_model(model, X_train_s, y_train, X_test_s, y_test, model_name=model_name)
            else:
                metrics = train_eval_model(model, X_train, y_train, X_test, y_test, model_name=model_name)


            row = {
                "model": model_name,
                "horizon": h,
                "threshold": th,
                "n_samples": len(df),
                "target_rate": df["target"].mean(),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
            results.append(row)

            rate_pos = (df["target"] == 1).mean()
            rate_neg = (df["target"] == -1).mean()
            rate_neu = (df["target"] == 0).mean()


            if verbose:
                print("=" * 60)
                print(f"model={model_name} | threshold={th} | horizon={h}")
                print(f"n={row['n_samples']} | "f"pos={rate_pos:.3f} | "f"neu={rate_neu:.3f} | "f"neg={rate_neg:.3f}")
                print(f"accuracy={row['accuracy']:.3f} | bal_acc={row['balanced_accuracy']:.3f}")
                print(metrics["confusion_matrix"])
                print(metrics["report"])

    return pd.DataFrame(results).sort_values(by="balanced_accuracy", ascending=False)


models = ["logreg", "knn", "rf","xgb"]
all_results = []

for m in models:
    df_res = test_thresholds(model_name=m, verbose=True)
    all_results.append(df_res)

final = pd.concat(all_results).sort_values("balanced_accuracy", ascending=False)
print(final.head(20))
final.to_csv("results.csv", index = False)


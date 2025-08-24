import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

MAX_SEQ_LENGTH = 20
HIGH_INTENT = {"purchase", "add_to_cart", "favorite"}

def label_and_balance(df):
    high = df[df["behavior_type"].isin(HIGH_INTENT)].copy()
    high["label"] = 1
    low_pool = df[df["behavior_type"] == "view"]
    low = low_pool.sample(n=len(high) * 3, replace=True, random_state=42).copy()
    low["label"] = 0
    train_df = pd.concat([high, low], ignore_index=True).sample(frac=1, random_state=42)
    return train_df

def encode_ids(train_df):
    encoders = {}
    for col in ["user_id", "product_id", "trader_id", "shop_id"]:
        encoder = LabelEncoder()
        values = train_df[col].astype(str).tolist()
        values.append("unknown")
        encoder.fit(values)
        train_df[f"{col}_encoded"] = encoder.transform(np.where(train_df[col].astype(str).isin(encoder.classes_), train_df[col].astype(str), "unknown"))
        encoders[col] = encoder
    return train_df, encoders

def one_hot_and_columns(train_df):
    train_df = pd.get_dummies(train_df, columns=["gender", "type", "category"], prefix=["gender", "type", "cat"])
    training_columns = train_df.drop("label", axis=1).columns
    joblib.dump(training_columns, "training_columns.joblib")
    return train_df, training_columns

def scale_and_time_features(train_df):
    price_scaler = MinMaxScaler()
    train_df["price_scaled"] = price_scaler.fit_transform(train_df[["price"]])
    train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24.0)
    train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24.0)
    train_df["day_sin"] = np.sin(2 * np.pi * train_df["day_of_week"] / 7.0)
    train_df["day_cos"] = np.cos(2 * np.pi * train_df["day_of_week"] / 7.0)
    joblib.dump(price_scaler, "price_scaler.joblib")
    return train_df

def prepare_sequences(train_df):
    train_df["date"] = pd.to_datetime(train_df["date"])
    train_df["timestamp"] = (train_df["date"] - train_df["date"].min()).dt.days
    train_df = train_df.sort_values(by=["user_id_encoded", "date"])
    def get_history(series, window=MAX_SEQ_LENGTH):
        hist = []
        out = []
        for v in series:
            out.append(hist[-window:].copy())
            hist.append(v)
        return out
    train_df["product_history"] = train_df.groupby("user_id_encoded")["product_id_encoded"].transform(lambda x: get_history(x) if len(x) > 0 else [[]] * len(x))
    train_df["timestamp_history"] = train_df.groupby("user_id_encoded")["timestamp"].transform(lambda x: get_history(x) if len(x) > 0 else [[]] * len(x))
    def time_diffs(row):
        cur = row["timestamp"]
        hist = row["timestamp_history"]
        return [cur - t for t in hist]
    train_df["time_diff_history"] = train_df.apply(lambda r: time_diffs(r) if len(r["timestamp_history"]) > 0 else [0] * MAX_SEQ_LENGTH, axis=1)
    return train_df

def engineer_advanced_features(train_df):
    pop = train_df["product_id_encoded"].value_counts().to_dict()
    train_df["product_popularity"] = train_df["product_id_encoded"].map(pop).fillna(0)
    act = train_df["user_id_encoded"].value_counts().to_dict()
    train_df["user_activity"] = train_df["user_id_encoded"].map(act).fillna(0)
    train_df["recency"] = train_df.groupby("user_id_encoded")["timestamp"].diff().fillna(0)
    for col in ["product_popularity", "user_activity", "recency"]:
        train_df[col] = np.clip(train_df[col], 0, None)
        train_df[col] = np.log1p(train_df[col])
    scalers = {}
    for col in ["product_popularity", "user_activity", "recency"]:
        sc = MinMaxScaler()
        train_df[col] = sc.fit_transform(train_df[[col]])
        scalers[col] = sc
    joblib.dump(scalers, "aux_scalers.joblib")
    return train_df

def split_data(train_df):
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(train_df, train_df["label"], test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return (X_train_full, y_train_full), (X_val, y_val), (X_test, y_test)

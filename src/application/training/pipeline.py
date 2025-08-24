import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json, os, joblib
from src.ml.fe_pipeline import label_and_balance, encode_ids, one_hot_and_columns, scale_and_time_features, prepare_sequences, engineer_advanced_features, split_data, MAX_SEQ_LENGTH
from src.ml.model_build import build_model
from src.ml.callbacks import DelayedEarlyStopping

def make_input_dict(df, history_padded, time_diff_padded):
    X = {
        "user_id": df["user_id_encoded"].values,
        "product_id": df["product_id_encoded"].values,
        "product_history": history_padded,
        "time_diff_history": time_diff_padded,
        "numerical_features": df[["price_scaled", "hour_sin", "hour_cos", "day_sin", "day_cos", "product_popularity", "user_activity", "recency"]].values,
        "categorical_features": df.filter(regex="gender_|type_|cat_").values,
    }
    return {k: (v if v.ndim > 1 else v.reshape(-1, 1)) for k, v in X.items()}

def run(df):
    train_df = label_and_balance(df)
    train_df, encoders = encode_ids(train_df)
    train_df, training_columns = one_hot_and_columns(train_df)
    train_df = scale_and_time_features(train_df)
    train_df = prepare_sequences(train_df)
    train_df = engineer_advanced_features(train_df)
    (X_train_full, y_train_full), (X_val, y_val), (X_test, y_test) = split_data(train_df)
    X_train_hist = pad_sequences(X_train_full["product_history"], maxlen=MAX_SEQ_LENGTH, padding="post", truncating="pre")
    X_val_hist = pad_sequences(X_val["product_history"], maxlen=MAX_SEQ_LENGTH, padding="post", truncating="pre")
    X_test_hist = pad_sequences(X_test["product_history"], maxlen=MAX_SEQ_LENGTH, padding="post", truncating="pre")
    X_train_time = pad_sequences(X_train_full["time_diff_history"], maxlen=MAX_SEQ_LENGTH, padding="post", truncating="pre", dtype="float32")
    X_val_time = pad_sequences(X_val["time_diff_history"], maxlen=MAX_SEQ_LENGTH, padding="post", truncating="pre", dtype="float32")
    X_test_time = pad_sequences(X_test["time_diff_history"], maxlen=MAX_SEQ_LENGTH, padding="post", truncating="pre", dtype="float32")
    X_train = make_input_dict(X_train_full, X_train_hist, X_train_time)
    X_val = make_input_dict(X_val, X_val_hist, X_val_time)
    X_test = make_input_dict(X_test, X_test_hist, X_test_time)
    num_users = len(encoders["user_id"].classes_)
    num_products = len(encoders["product_id"].classes_)
    num_cats = X_train["categorical_features"].shape[1]
    model = build_model(num_users, num_products, num_cats, max_seq_len=MAX_SEQ_LENGTH)
    early = DelayedEarlyStopping()
    model.fit(X_train, y_train_full, validation_data=(X_val, y_val), epochs=100, batch_size=128, callbacks=[early], verbose=1)
    os.makedirs("artifacts", exist_ok=True)
    model.save("final_recommendation_model.keras")
    joblib.dump(encoders, "encoders.joblib")
    joblib.dump(training_columns, "training_columns.joblib")
    test_results = model.evaluate(X_test, y_test, batch_size=128, return_dict=True)
    y_pred_prob = model.predict(X_test, batch_size=128).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    if "auc" not in test_results:
        test_results["auc"] = float(roc_auc_score(y_test, y_pred_prob))
    with open("artifacts/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": float(test_results.get("loss")),
                "auc": float(test_results.get("auc")),
                "report": classification_report(y_test, y_pred, digits=4, zero_division=0, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return model, encoders, training_columns, test_results

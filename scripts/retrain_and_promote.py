import os, json, shutil
from datetime import datetime
import pandas as pd
from src.infrastructure.db import engine
from src.application.training.pipeline import run

THRESHOLD_AUC = 0.80
ARTS = ["final_recommendation_model.keras","encoders.joblib","training_columns.joblib","price_scaler.joblib","aux_scalers.joblib"]

def load_df():
    q = """
    SELECT 
        b.user_id,
        u.gender,
        b.product_id,
        b.behavior_type,
        b.behavior_time,
        p.name AS product_name,
        p.category,
        p.type,
        p.price,
        p.trader_id,
        t.shop_id
    FROM Behavior b
    JOIN Product p ON b.product_id = p.product_id
    JOIN User u ON b.user_id = u.user_id
    JOIN Trader t ON p.trader_id = t.user_id
    """
    df = pd.read_sql(q, engine)
    df["behavior_time"] = pd.to_datetime(df["behavior_time"])
    df["date"] = df["behavior_time"].dt.date
    df["hour"] = df["behavior_time"].dt.hour
    df["day_of_week"] = df["behavior_time"].dt.dayofweek
    df.drop_duplicates(inplace=True)
    df = df[df["price"] > 0]
    return df

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    df = load_df()
    model, encoders, training_columns, test_results = run(df)
    auc = float(test_results.get("auc")) if "auc" in test_results else 0.0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join("models", stamp)
    os.makedirs(version_dir, exist_ok=True)
    for f in ARTS:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(version_dir, f))
    metrics_path = os.path.join(version_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"auc": auc, "loss": float(test_results.get("loss", 0.0))}, f, ensure_ascii=False, indent=2)
    if auc >= THRESHOLD_AUC:
        for f in ARTS:
            src = os.path.join(version_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, f)
        with open("active_model.json", "w", encoding="utf-8") as f:
            json.dump({"active_version": stamp}, f)
        print("PROMOTED", stamp, "AUC", auc)
    else:
        print("REJECTED", stamp, "AUC", auc)

if __name__ == "__main__":
    main()

import os
import pandas as pd
from src.infrastructure.db import engine
from src.application.training.pipeline import run

os.makedirs("artifacts", exist_ok=True)
query = """
SELECT 
    b.user_id,
    u.gender,
    b.product_id,
    b.behavior_type,
    p.name AS product_name,
    p.category,
    p.type,
    p.price,
    p.trader_id,
    t.shop_id,
    DATE(b.behavior_time) AS date,
    HOUR(b.behavior_time) AS hour,
    (DAYOFWEEK(b.behavior_time) - 1) AS day_of_week
FROM Behavior b
JOIN Product p ON b.product_id = p.product_id
JOIN User u ON b.user_id = u.user_id
JOIN Trader t ON p.trader_id = t.user_id
"""
df = pd.read_sql(query, engine)
df["date"] = pd.to_datetime(df["date"])
df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0).astype(int)
run(df)

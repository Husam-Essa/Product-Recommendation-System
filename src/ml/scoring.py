import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.infrastructure.artifacts import model, encoders, training_columns, price_scaler, aux_scalers
from src.domain.state import api_data
from src.ml.fe_pipeline import MAX_SEQ_LENGTH

def scale_price_value(price: float) -> float:
    return float(price_scaler.transform(pd.DataFrame({"price": [price]}))[0][0])

def encode_with_unknown(encoder, value):
    s = str(value)
    if s in encoder.classes_:
        return int(encoder.transform([s])[0])
    return int(encoder.transform(["unknown"])[0])

def get_user_gender(user_id):
    user = api_data["users"].get(user_id)
    return user.get("gender") if user else None

def get_user_history(user_id):
    user_behaviors = [b for b in api_data["behaviors"] if b["user_id"] == user_id]
    if not user_behaviors:
        return [], []
    parsed = []
    for b in user_behaviors:
        ts_raw = b.get("behavior_time")
        t = None
        try:
            t = pd.to_datetime(ts_raw, utc=True)
        except Exception:
            try:
                t = pd.to_datetime(str(ts_raw).replace("Z", "+00:00"), utc=True)
            except Exception:
                t = None
        parsed.append((b, t))
    parsed = [pt for pt in parsed if pd.notnull(pt[1])]
    if not parsed:
        return [], []
    parsed.sort(key=lambda x: x[1])
    products = [b["product_id"] for b, _ in parsed]
    timestamps = [t.to_pydatetime() for _, t in parsed]
    min_time = min(timestamps)
    ts = [(t - min_time).days for t in timestamps]
    return products, ts

def build_history_inputs(products, ts):
    if not products:
        ph = [[0] * MAX_SEQ_LENGTH]
        td = [[0] * MAX_SEQ_LENGTH]
        return np.array(ph), np.array(td, dtype="float32")
    current_ts = (max(ts) if ts else 0) + 1
    diffs = [current_ts - t for t in ts]
    enc_products = [encode_with_unknown(encoders["product_id"], p) for p in products]
    ph = pad_sequences([enc_products], maxlen=MAX_SEQ_LENGTH, padding="post", truncating="pre")
    td = pad_sequences([diffs], maxlen=MAX_SEQ_LENGTH, padding="post", truncating="pre", dtype="float32")
    return ph, td

def get_candidates(limit: int):
    price_values = []
    for p in api_data["products"].values():
        pr = p.get("price")
        try:
            if pr is not None:
                price_values.append(float(pr))
        except Exception:
            pass
    median_price = float(np.median(price_values)) if price_values else None
    df_data = []
    for p in api_data["products"].values():
        if p.get("is_available") is False:
            continue
        pid = p.get("product_id")
        if pid is None:
            continue
        price_value = None
        try:
            price_value = float(p.get("price")) if p.get("price") is not None else None
        except Exception:
            price_value = None
        if price_value is None:
            if median_price is None:
                continue
            price_value = median_price
        ptype = p.get("type")
        if ptype is None:
            ptype = "physical"
        pcat = p.get("category")
        if pcat is None:
            pcat = "misc"
        try:
            df_data.append({"product_id": int(pid), "price": float(price_value), "type": str(ptype), "category": str(pcat)})
        except Exception:
            continue
    if limit and len(df_data) > limit:
        df_data = df_data[:limit]
    return pd.DataFrame(df_data)

def categorical_vector(user_gender, prod_type, prod_cat):
    cat_cols = [c for c in training_columns if c.startswith(("gender_", "type_", "cat_"))]
    vec = np.zeros((len(cat_cols),), dtype="float32")
    def _hot(prefix, raw):
        if raw is None:
            return
        raw_s = str(raw).strip().lower()
        candidates = []
        if prefix == "gender_":
            if raw_s in {"0", "male", "m"}:
                candidates = ["male", "0", "m"]
            elif raw_s in {"1", "female", "f"}:
                candidates = ["female", "1", "f"]
            else:
                candidates = [raw_s]
        elif prefix == "type_":
            if raw_s in {"1", "true", "digital", "yes"}:
                candidates = ["digital", "1", "true"]
            elif raw_s in {"0", "false", "physical", "no"}:
                candidates = ["physical", "0", "false"]
            else:
                candidates = [raw_s]
        elif prefix == "cat_":
            candidates = [raw_s, str(raw).strip()]
        for cand in candidates:
            name = f"{prefix}{cand}"
            if name in cat_cols:
                vec[cat_cols.index(name)] = 1.0
                return
    _hot("gender_", user_gender)
    _hot("type_", prod_type)
    _hot("cat_", prod_cat)
    return vec

def compute_online_adv(user_id, product_id, ts):
    product_behaviors = [b for b in api_data["behaviors"] if b["product_id"] == product_id]
    pop = len(product_behaviors)
    user_behaviors = [b for b in api_data["behaviors"] if b["user_id"] == user_id]
    act = len(user_behaviors)
    last_diff = 0
    if ts:
        current_ts = max(ts) + 1
        last_diff = current_ts - max(ts)
    vals = {"product_popularity": np.log1p(max(0, int(pop))), "user_activity": np.log1p(max(0, int(act))), "recency": np.log1p(max(0, int(last_diff)))}
    for k in list(vals.keys()):
        sc = aux_scalers.get(k)
        vals[k] = float(sc.transform(np.array(vals[k]).reshape(-1, 1))[0][0]) if sc is not None else 0.0
    return vals

def recommend_for_user(user_id_raw: int, top_k=10, candidate_limit=200):
    user_gender = get_user_gender(user_id_raw)
    hist_products, ts = get_user_history(user_id_raw)
    ph, td = build_history_inputs(hist_products, ts)
    candidates = get_candidates(candidate_limit)
    user_enc = encode_with_unknown(encoders["user_id"], user_id_raw)
    user_arr = np.array([[user_enc]])
    now = pd.Timestamp.now()
    hour_sin = np.sin(2 * np.pi * now.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * now.hour / 24.0)
    day_sin = np.sin(2 * np.pi * now.dayofweek / 7.0)
    day_cos = np.cos(2 * np.pi * now.dayofweek / 7.0)
    recs = []
    for _, row in candidates.iterrows():
        pid = int(row["product_id"])
        p_enc = encode_with_unknown(encoders["product_id"], pid)
        price = float(row["price"])
        ptype = str(row["type"])
        pcat = str(row["category"])
        price_scaled = scale_price_value(price)
        adv = compute_online_adv(user_id_raw, pid, ts)
        numerical = np.array([[price_scaled, hour_sin, hour_cos, day_sin, day_cos, adv["product_popularity"], adv["user_activity"], adv["recency"]]], dtype="float32")
        cat_vec = categorical_vector(user_gender, ptype, pcat).reshape(1, -1)
        input_dict = {"user_id": user_arr, "product_id": np.array([[p_enc]]), "product_history": ph, "time_diff_history": td, "numerical_features": numerical, "categorical_features": cat_vec}
        prob = float(model.predict(input_dict, verbose=0)[0][0])
        recs.append((pid, prob))
    recs.sort(key=lambda x: x[1], reverse=True)
    top = recs[:top_k]
    if not top:
        return []
    info_map = {}
    for pid, _ in top:
        product = api_data["products"].get(pid)
        if product:
            info_map[pid] = (product.get("name", ""), product.get("category", ""))
    return [{"product_id": int(pid), "product_name": info_map.get(int(pid), ("", ""))[0], "category": info_map.get(int(pid), ("", ""))[1], "score": float(score)} for pid, score in top]

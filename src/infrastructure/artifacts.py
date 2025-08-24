import joblib
import tensorflow as tf
from src.ml.custom_layers import TimeDecayAttentionLayer

model = None
encoders = None
training_columns = None
price_scaler = None
aux_scalers = {}

def reload_artifacts():
    global model, encoders, training_columns, price_scaler, aux_scalers
    model = tf.keras.models.load_model(
        "final_recommendation_model.keras",
        custom_objects={"TimeDecayAttentionLayer": TimeDecayAttentionLayer}
    )
    encoders = joblib.load("encoders.joblib")
    training_columns = joblib.load("training_columns.joblib")
    price_scaler = joblib.load("price_scaler.joblib")
    try:
        aux_scalers = joblib.load("aux_scalers.joblib")
    except:
        aux_scalers = {}

reload_artifacts()

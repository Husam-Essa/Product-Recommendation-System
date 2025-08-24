import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, Dot
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from src.ml.custom_layers import TimeDecayAttentionLayer

def build_model(num_users, num_products, num_categorical_features, embedding_size=128, l2_strength=0.01, learning_rate=0.005, dense_1_units=128, dense_2_units=64, dropout_1_rate=0.3, max_seq_len=20):
    input_user = Input(shape=(1,), name="user_id")
    input_product = Input(shape=(1,), name="product_id")
    input_history = Input(shape=(max_seq_len,), name="product_history")
    input_time_diff = Input(shape=(max_seq_len,), name="time_diff_history")
    input_numerical = Input(shape=(8,), name="numerical_features")
    input_categorical = Input(shape=(num_categorical_features,), name="categorical_features")
    user_embedding_layer = Embedding(input_dim=num_users, output_dim=embedding_size, name="user_embedding")
    product_embedding_layer = Embedding(input_dim=num_products, output_dim=embedding_size, name="product_embedding", mask_zero=True)
    user_vec = Flatten()(user_embedding_layer(input_user))
    product_vec = Flatten()(product_embedding_layer(input_product))
    history_embedding = product_embedding_layer(input_history)
    history_vec = TimeDecayAttentionLayer(name="history_attention")([history_embedding, input_time_diff, input_history])
    interaction_dot = Dot(axes=1, name="user_product_interaction")([user_vec, product_vec])
    merged = Concatenate()([user_vec, product_vec, history_vec, interaction_dot, input_numerical, input_categorical])
    x = Dense(dense_1_units, activation="relu", kernel_regularizer=regularizers.l2(l2_strength))(merged)
    x = Dropout(dropout_1_rate)(x)
    x = Dense(dense_2_units, activation="relu", kernel_regularizer=regularizers.l2(l2_strength))(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_user, input_product, input_history, input_time_diff, input_numerical, input_categorical], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model

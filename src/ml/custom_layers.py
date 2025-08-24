import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class TimeDecayAttentionLayer(Layer):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.attention_scorer = Dense(1, activation="tanh", name="attention_scorer")

    def call(self, inputs):
        history_embedding, time_diff_history, history_ids = inputs
        scores = self.attention_scorer(history_embedding)
        time_diff = tf.expand_dims(tf.cast(time_diff_history, tf.float32), -1)
        decay = tf.exp(-self.alpha * time_diff)
        weighted = scores * decay
        mask = tf.cast(tf.not_equal(history_ids, 0), tf.float32)
        mask = tf.expand_dims(mask, -1)
        weighted = weighted + (1.0 - mask) * (-1e9)
        attn = tf.nn.softmax(weighted, axis=1)
        context = tf.reduce_sum(history_embedding * attn, axis=1)
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha})
        return cfg

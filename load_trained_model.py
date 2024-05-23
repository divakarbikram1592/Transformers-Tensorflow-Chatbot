import tensorflow as tf
from positional_encoding import PositionalEncoding
from multihead_attention_layer import MultiHeadAttentionLayer


class LoadTrainedModel:

    def __init__(self):
        print("loading trained model...")

    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        # (batch_size, 1, 1, sequence length)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def get_trained_model(self):
        filename = "chatbot_model.h5"
        loaded_model = tf.keras.models.load_model(
            filename,
            custom_objects={
                "PositionalEncoding": PositionalEncoding,
                "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
                "create_padding_mask": self.create_padding_mask,
            },
            compile=False,
        )

        return loaded_model

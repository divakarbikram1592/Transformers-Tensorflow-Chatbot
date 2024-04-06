import tensorflow as tf

from load_dataset import LoadDataset


class LoadTrainedModel:

    def __init__(self):
        print("Loading trained model...")
        self.questions, self.answers = LoadDataset().load_conversations()

    def create_padding_mask(self, x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        # (batch_size, 1, 1, sequence length)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def scaled_dot_product_attention(self, query, key, value, mask):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(query, key, transpose_b=True)

        # scale matmul_qk
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        # add the mask to zero out padding tokens
        if mask is not None:
            logits += mask * -1e9

        # softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(logits, axis=-1)

        output = tf.matmul(attention_weights, value)

        return output

    class PositionalEncoding(tf.keras.layers.Layer):
        def __init__(self, position, d_model, **kwargs):
            super().__init__(**kwargs)

            self.position = position
            self.d_model = d_model
            self.pos_encoding = self.positional_encoding(position, d_model)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "position": self.position,
                    "d_model": self.d_model,
                }
            )
            return config

        def get_angles(self, position, i, d_model):
            angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            return position * angles

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(
                position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                d_model=d_model,
            )
            # apply sin to even index in the array
            sines = tf.math.sin(angle_rads[:, 0::2])
            # apply cos to odd index in the array
            cosines = tf.math.cos(angle_rads[:, 1::2])

            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[tf.newaxis, ...]
            return tf.cast(pos_encoding, tf.float32)

        def call(self, inputs):
            return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

    class MultiHeadAttentionLayer(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, **kwargs):
            # assert d_model % num_heads == 0
            super().__init__(**kwargs)
            self.num_heads = num_heads
            self.d_model = d_model

            self.depth = d_model // self.num_heads

            self.query_dense = tf.keras.layers.Dense(units=d_model)
            self.key_dense = tf.keras.layers.Dense(units=d_model)
            self.value_dense = tf.keras.layers.Dense(units=d_model)

            self.dense = tf.keras.layers.Dense(units=d_model)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "num_heads": self.num_heads,
                    "d_model": self.d_model,
                }
            )
            return config

        def split_heads(self, inputs, batch_size):
            inputs = tf.keras.layers.Lambda(
                lambda inputs: tf.reshape(
                    inputs, shape=(batch_size, -1, self.num_heads, self.depth)
                )
            )(inputs)
            return tf.keras.layers.Lambda(
                lambda inputs: tf.transpose(inputs, perm=[0, 2, 1, 3])
            )(inputs)

        def call(self, inputs):
            query, key, value, mask = (
                inputs["query"],
                inputs["key"],
                inputs["value"],
                inputs["mask"],
            )
            batch_size = tf.shape(query)[0]

            # linear layers
            query = self.query_dense(query)
            key = self.key_dense(key)
            value = self.value_dense(value)

            # split heads
            query = self.split_heads(query, batch_size)
            key = self.split_heads(key, batch_size)
            value = self.split_heads(value, batch_size)

            # scaled dot-product attention
            scaled_attention = LoadTrainedModel().scaled_dot_product_attention(query, key, value, mask)
            scaled_attention = tf.keras.layers.Lambda(
                lambda scaled_attention: tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            )(scaled_attention)

            # concatenation of heads
            concat_attention = tf.keras.layers.Lambda(
                lambda scaled_attention: tf.reshape(
                    scaled_attention, (batch_size, -1, self.d_model)
                )
            )(scaled_attention)

            # final linear layer
            outputs = self.dense(concat_attention)

            return outputs

    def get_trained_model(self):
        filename = "chatbot_model.h5"
        loaded_model = tf.keras.models.load_model(
            filename,
            custom_objects={
                "PositionalEncoding": self.PositionalEncoding,
                "MultiHeadAttentionLayer": self.MultiHeadAttentionLayer,
            },
            compile=False,
        )

        return loaded_model

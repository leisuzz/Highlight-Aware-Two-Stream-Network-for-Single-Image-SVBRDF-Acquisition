import tensorflow as tf
from tensorflow.keras.activations import softmax

class SoftAttention(tf.keras.layers.Layer):
    def __init__(self, intermediate_fc_units_count, dropout_rate,
                 soft_attention_output_units=1):
        self._soft_attention_output_units = soft_attention_output_units
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(intermediate_fc_units_count, activation='relu'),
            # (batch_size, seq_len, dff)
            tf.keras.layers.Dropout(dropout_rate),  # (batch_size, seq_len, d_model)
            tf.keras.layers.Dense(soft_attention_output_units, lambda x: softmax(x, axis=1)),
            # (batch_size, seq_len, d_model)
        ])

        self._softmax_activation = tf.keras.layers.Activation(activation='softmax')
        super(SoftAttention, self).__init__()

    def call(self, x):
        # x = (batch, seq_len, d_model)
        # attention = (batch, seq_len, 1)
        attention = self.mlp(x)
        att_list = []
        for i in range(self._soft_attention_output_units):
            att_list.append(
                tf.math.reduce_sum(attention[:, :, i: i + 1] * x, axis=1)
            )

        # return tf.concat(att_list, axis=1)
        return tf.stack(att_list, axis=1)
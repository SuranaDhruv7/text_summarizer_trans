
import tensorflow as tf

# ============================================================
# 1. POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, d_model, max_length=2048):
        super().__init__()

        positions = tf.cast(
            tf.range(max_length)[:, tf.newaxis],
            tf.float32
        )

        dimensions = tf.cast(
            tf.range(d_model)[tf.newaxis, :],
            tf.float32
        )

        angle_rates = 1.0 / tf.pow(
            10000.0,
            (2.0 * tf.floor(dimensions / 2.0)) / d_model
        )

        angles = positions * angle_rates

        even_mask = tf.equal(
            tf.math.floormod(tf.range(d_model), 2),
            0
        )[tf.newaxis, :]

        encoding = tf.where(
            even_mask,
            tf.sin(angles),
            tf.cos(angles)
        )

        self.pos_encoding = encoding[tf.newaxis, :, :]

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]

        return inputs + tf.cast(
            self.pos_encoding[:, :seq_length, :],
            inputs.dtype
        )


# ============================================================
# 2. SCALED DOT-PRODUCT ATTENTION
# ============================================================

def scaled_dot_product_attention(q, k, v, mask=None):

    depth = tf.cast(tf.shape(k)[-1], q.dtype)

    # Q multiplied by transpose of K
    scores = tf.matmul(q, k, transpose_b=True)

    # Scale attention scores
    scores = scores / tf.sqrt(depth)

    # Mask padding tokens or future tokens
    if mask is not None:
        mask = tf.cast(mask, tf.bool)

        scores = tf.where(
            mask,
            tf.cast(-1e9, scores.dtype),
            scores
        )

    # Convert scores into attention probabilities
    attention_weights = tf.nn.softmax(scores, axis=-1)

    # Weighted combination of values
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


# ============================================================
# 3. MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.output_dense = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, x):

        shape = tf.shape(x)

        x = tf.reshape(
            x,
            [
                shape[0],
                shape[1],
                self.num_heads,
                self.depth
            ]
        )

        return tf.transpose(x, [0, 2, 1, 3])

    def call(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=False
    ):

        q = self.split_heads(self.wq(query))
        k = self.split_heads(self.wk(key))
        v = self.split_heads(self.wv(value))

        attention_output, attention_weights = (
            scaled_dot_product_attention(
                q, k, v, attention_mask
            )
        )

        attention_output = self.dropout(
            attention_output,
            training=training
        )

        # Combine all attention heads
        attention_output = tf.transpose(
            attention_output,
            [0, 2, 1, 3]
        )

        shape = tf.shape(attention_output)

        attention_output = tf.reshape(
            attention_output,
            [shape[0], shape[1], self.d_model]
        )

        output = self.output_dense(attention_output)

        return output, attention_weights


# ============================================================
# 4. FEED-FORWARD NETWORK
# ============================================================

class FeedForwardNetwork(tf.keras.layers.Layer):

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(
            d_ff,
            activation="relu"
        )

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, inputs, training=False):

        x = self.dense1(inputs)
        x = self.dropout(x, training=training)

        return self.dense2(x)


# ============================================================
# 5. ENCODER LAYER
# ============================================================

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout_rate=0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model, num_heads, dropout_rate
        )

        self.ffn = FeedForwardNetwork(
            d_model, d_ff, dropout_rate
        )

        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, src_mask=None, training=False):

        # Self-attention: query, key and value come from x
        attention_output, _ = self.attention(
            x,
            x,
            x,
            attention_mask=src_mask,
            training=training
        )

        # Residual connection and normalization
        x = self.norm1(
            x + self.drop1(attention_output, training=training)
        )

        # Feed-forward network
        ffn_output = self.ffn(x, training=training)

        # Second residual connection and normalization
        x = self.norm2(
            x + self.drop2(ffn_output, training=training)
        )

        return x


# ============================================================
# 6. DECODER LAYER
# ============================================================

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout_rate=0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout_rate
        )

        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, dropout_rate
        )

        self.ffn = FeedForwardNetwork(
            d_model, d_ff, dropout_rate
        )

        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)
        self.drop3 = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        x,
        encoder_output,
        self_mask=None,
        src_mask=None,
        training=False
    ):

        # 1. Masked self-attention
        attention1, _ = self.self_attention(
            x,
            x,
            x,
            attention_mask=self_mask,
            training=training
        )

        x = self.norm1(
            x + self.drop1(attention1, training=training)
        )

        # 2. Cross-attention: decoder attends to encoder output
        attention2, _ = self.cross_attention(
            x,
            encoder_output,
            encoder_output,
            attention_mask=src_mask,
            training=training
        )

        x = self.norm2(
            x + self.drop2(attention2, training=training)
        )

        # 3. Feed-forward network
        ffn_output = self.ffn(x, training=training)

        x = self.norm3(
            x + self.drop3(ffn_output, training=training)
        )

        return x


# ============================================================
# 7. COMPLETE ENCODER-DECODER TRANSFORMER
# ============================================================

class Transformer(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_source_length=615,
        max_target_length=64,
        pad_id=0,
        dropout_rate=0.1
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # Token embeddings
        self.encoder_embedding = tf.keras.layers.Embedding(
            vocab_size, d_model
        )

        self.decoder_embedding = tf.keras.layers.Embedding(
            vocab_size, d_model
        )

        # Positional encoding
        self.encoder_position = PositionalEncoding(
            d_model, max_source_length
        )

        self.decoder_position = PositionalEncoding(
            d_model, max_target_length
        )

        self.encoder_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_dropout = tf.keras.layers.Dropout(dropout_rate)

        # Stack encoder layers
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_encoder_layers)
        ]

        # Stack decoder layers
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_decoder_layers)
        ]

        # Produce a score for each vocabulary token
        self.output_projection = tf.keras.layers.Dense(vocab_size)

    # Padding mask: True means the token is blocked
    def create_padding_mask(self, token_ids):

        return tf.equal(
            token_ids[:, tf.newaxis, tf.newaxis, :],
            self.pad_id
        )

    # Prevent decoder positions from looking at future tokens
    @staticmethod
    def create_look_ahead_mask(length):

        mask = tf.linalg.band_part(
            tf.ones([length, length], dtype=tf.int32),
            -1,
            0
        )

        return tf.equal(mask, 0)

    # ---------------- ENCODER ----------------

    def encode(self, article_ids, training=False):

        src_mask = self.create_padding_mask(article_ids)

        x = self.encoder_embedding(article_ids)

        x *= tf.math.sqrt(tf.cast(self.d_model, x.dtype))

        x = self.encoder_position(x)

        x = self.encoder_dropout(x, training=training)

        for layer in self.encoder_layers:
            x = layer(
                x,
                src_mask=src_mask,
                training=training
            )

        return x, src_mask

    # ---------------- DECODER ----------------

    def decode(
        self,
        decoder_ids,
        encoder_output,
        src_mask,
        training=False
    ):

        tgt_padding_mask = self.create_padding_mask(decoder_ids)

        target_length = tf.shape(decoder_ids)[1]

        causal_mask = self.create_look_ahead_mask(target_length)

        causal_mask = causal_mask[
            tf.newaxis, tf.newaxis, :, :
        ]

        # Block both future positions and padded target keys
        combined_mask = tf.logical_or(
            causal_mask,
            tgt_padding_mask
        )

        x = self.decoder_embedding(decoder_ids)

        x *= tf.math.sqrt(tf.cast(self.d_model, x.dtype))

        x = self.decoder_position(x)

        x = self.decoder_dropout(x, training=training)

        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_output,
                self_mask=combined_mask,
                src_mask=src_mask,
                training=training
            )

        return self.output_projection(x)

    # ---------------- FORWARD PASS ----------------

    def call(self, inputs, training=False):

        article_ids, decoder_ids = inputs

        encoder_output, src_mask = self.encode(
            article_ids,
            training=training
        )

        logits = self.decode(
            decoder_ids,
            encoder_output,
            src_mask,
            training=training
        )

        return logits
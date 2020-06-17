import tensorflow as tf


class GetTransformerModel:
    def __init__(self, stack_num=2, heads_num=8, drop_rate=0.1, d_model=512, feed_forward_hidden_units_num=2048,
                 sequence_length=128, gpu_num=1):
        self.stack_num = stack_num
        self.heads_num = heads_num
        self.drop_rate = drop_rate
        self.d_model = d_model
        self.feed_forward_hidden_units_num = feed_forward_hidden_units_num
        self.sequence_length = sequence_length
        self.gpu_num = gpu_num

    def set(self, stack_num=None, heads_num=None, drop_rate=None, d_model=None, feed_forward_hidden_units_num=None,
            sequence_length=None, gpu_num=None):
        if stack_num is not None:
            self.stack_num = stack_num
        if heads_num is not None:
            self.heads_num = heads_num
        if drop_rate is not None:
            self.drop_rate = drop_rate
        if d_model is not None:
            self.d_model = d_model
        if feed_forward_hidden_units_num is not None:
            self.feed_forward_hidden_units_num = feed_forward_hidden_units_num
        if sequence_length is not None:
            self.sequence_length = sequence_length
        if gpu_num is not None:
            self.gpu_num = gpu_num

    @staticmethod
    def layer_normalization(inputs, epsilon=1e-8, scope="layer_normalization"):
        """
        Applies layer normalization. See https://arxiv.org/abs/1607.06450.
        inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** .5)
            outputs = gamma * normalized + beta

        return outputs

    @staticmethod
    def mask(inputs, queries=None, keys=None, mask_type=None, scope='mask'):
        """
        Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (N, T_q, T_k)
        queries: 3d tensor. (N, T_q, d)
        keys: 3d tensor. (N, T_k, d)
        where T_q means query_token_length
        and T_k means key_token_length
        these two value may be different
        e.g.,
        >> queries = tf.constant([[[1.],
                                   [2.],
                                   [0.]]],
                                   tf.float32) # (1, 3, 1)

        >> keys = tf.constant([[[4.],
                                [0.]]],
                                tf.float32)# (1, 2, 1)

        >> inputs = tf.constant([[[4., 0.],
                                  [8., 0.],
                                  [0., 0.]]], tf.float32)

        >> mask(inputs, queries, keys, "key")
        array([[[ 4.0000000e+00, -4.2949673e+09],
                [ 8.0000000e+00, -4.2949673e+09],
                [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)

        >> inputs = tf.constant([[[1., 0.],
                                  [1., 0.],
                                  [1., 0.]]], tf.float32)

        >> mask(inputs, queries, keys, "query")
        array([[[1., 0.],
                [1., 0.],
                [0., 0.]]], dtype=float32)
        """
        outputs = None
        padding_num = -2 ** 32 + 1

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if mask_type in ("k", "key", "keys"):
                # Generate masks
                masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
                masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
                masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

                # Apply masks to inputs
                # tf.ones_like(inputs)：按照inputs的形状创建一个将所有元素都设置为1的张量.
                paddings = tf.ones_like(inputs) * padding_num
                outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)

            elif mask_type in ("q", "query", "queries"):
                # Generate masks
                masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
                masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
                masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

                # Apply masks to inputs
                outputs = inputs * masks

            elif mask_type in ("f", "future", "right"):
                diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

                paddings = tf.ones_like(masks) * padding_num
                outputs = tf.where(tf.equal(masks, 0), paddings, inputs)

            else:
                print("Check if you entered type correctly!")

        return outputs

    def scaled_dot_product_attention(self, q, k, v, drop_rate=None, causality=False, training=True,
                                     attention_length=None, scope="scaled_dot_product_attention"):
        """
        See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.

        where T_q means query_token_length
        and T_k means key_token_length
        these two value can be different

        what's more, d_k means query_vector_size
        and d_v means value_vector_size
        these two value also can be different
        """
        if drop_rate is None:
            drop_rate = self.drop_rate

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(q, tf.transpose(k, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= d_k ** 0.5

            # key masking
            outputs = self.mask(outputs, q, k, mask_type="key")

            # causality or future blinding masking
            if causality:
                outputs = self.mask(outputs, mask_type="future")

            # softmax
            outputs = tf.nn.softmax(outputs)  # (N, T_q, T_k)

            # query masking
            outputs = self.mask(outputs, q, k, mask_type="query")  # (N, T_q, T_k)

            # dropout
            outputs = tf.layers.dropout(outputs, rate=drop_rate, training=training)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, v)  # (N, T_q, d_v)

        return outputs

    def multihead_attention(self, queries, keys, values, gpu_num=None, num_heads=None, dropout_rate=None,
                            training=True, causality=False, residual_connection=True, normalize=True, use_bias=True,
                            scope="multihead_attention"):
        """
        Applies multihead attention. See 3.2.2
        queries: A 3d tensor with shape of [N, T_q, d_model].
        keys: A 3d tensor with shape of [N, T_k, d_model].
        values: A 3d tensor with shape of [N, T_k, d_model].
        num_heads: An int. Number of heads.
        dropout_rate: A floating point number.
        training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked.
        scope: Optional scope for `variable_scope`.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        """
        if dropout_rate is None:
            dropout_rate = self.drop_rate
        if gpu_num is None:
            gpu_num = self.gpu_num
        if num_heads is None:
            num_heads = self.heads_num

        d_model = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            q = tf.layers.dense(queries, d_model, use_bias=use_bias)  # (N, T_q, d_model)
            k = tf.layers.dense(keys, d_model, use_bias=use_bias)  # (N, T_k, d_model)
            v = tf.layers.dense(values, d_model, use_bias=use_bias)  # (N, T_k, d_model)

            # Split and concat
            q = tf.concat(tf.split(q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
            k = tf.concat(tf.split(k, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            v = tf.concat(tf.split(v, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

            # Attention
            outputs = self.scaled_dot_product_attention(q, k, v,
                                                        drop_rate=dropout_rate,
                                                        causality=causality,
                                                        training=training)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

            if residual_connection:
                # Residual connection
                outputs += queries
            if normalize:
                # Normalize
                outputs = self.layer_normalization(outputs)

        return outputs

    def position_wise_feed_forward(self, inputs, hidden_units_num=None, scope="position_wise_feed_forward"):
        """
        position-wise feed forward net. See 3.3

        inputs: A 3d tensor with shape of [N, T, C].
        hidden_units_num: Hidden units number in position wise feed forward layer.
        scope: Optional scope for `variable_scope`.
        Returns:
          A 3d tensor with the same shape and dtype as inputs
        """
        if hidden_units_num is None:
            hidden_units_num = self.feed_forward_hidden_units_num

        outputs_length = inputs.get_shape().as_list()[-1]

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, hidden_units_num, activation=tf.nn.relu)

            # Outer layer
            outputs = tf.layers.dense(outputs, outputs_length)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.layer_normalization(outputs)

        return outputs

    def embedding(self, input_sequence, vocab_size, vector_size=None, use_pad=True, initializer_range=0.02, scope='embedding'):
        if vector_size is None:
            vector_size = self.d_model

        # 创建一个embedding矩阵，这里将第0维填充为0，用来对应各sequence中的[pad]标签
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable('embedding_matrix',
                                               dtype=tf.float32,
                                               shape=(vocab_size, vector_size),
                                               initializer=tf.truncated_normal_initializer(stddev=initializer_range))
            if use_pad:
                embedding_matrix = tf.concat((tf.zeros(shape=[1, vector_size]), embedding_matrix[1:, :]), 0)
            # embedding_matrix = tf.tanh(embedding_matrix)
            output = tf.nn.embedding_lookup(embedding_matrix, input_sequence)
        return output

    def position_embedding(self, inputs, max_len=None, vector_size=None, masking=True, scope="positional_embedding"):
        """
        Sinusoidal Positional_Encoding. See 3.5
        inputs: 3d tensor. (N, T, E)
        maxlen: scalar. Must be >= T
        masking: Boolean. If True, padding positions are set to zeros.
        scope: Optional scope for `variable_scope`.
        returns
        3d tensor that has the same shape as inputs.
        """
        if max_len is None:
            max_len = self.sequence_length
        if vector_size is None:
            vector_size = self.d_model

        n, t = tf.shape(inputs)[0], tf.shape(inputs)[1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(t), 0), [n, 1])  # (N, T)

            outputs = self.embedding(position_ind, max_len, use_pad=False, vector_size=vector_size)

            # masks
            if masking:
                zeros = tf.zeros_like(outputs)
                mask = tf.equal(tf.reduce_sum(tf.abs(inputs), axis=-1, keepdims=True), 0)
                mask = tf.tile(mask, [1, 1, vector_size])
                outputs = tf.where(mask, zeros, outputs)

        return tf.to_float(outputs)

    def encoder(self, inputs, training=True, scope='encoder'):
        """
        :param scope:
        :param training:boolean
        :param inputs:a embedding vector of input sequence, [batch_size, sequence_length, d_model]
        :return:a representation vector of input sequence, [batch_size, sequence_length, d_model]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # scale inputs
            outputs = inputs * self.d_model ** 0.5

            # position embedding
            outputs += self.position_embedding(outputs)
            outputs = tf.layers.dropout(outputs, self.drop_rate, training=training)

            # Stack Blocks
            for i in range(self.stack_num):
                with tf.variable_scope("block_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    outputs = self.multihead_attention(outputs, outputs, outputs, training=training)
                    # feed forward
                    outputs = self.position_wise_feed_forward(outputs)

        return outputs

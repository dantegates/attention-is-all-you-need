"""
Implementation of Transformer model, as described here

    https://arxiv.org/pdf/1706.03762.pdf
"""

import argparse
import logging
import subprocess as sp

import keras
import numpy as np
from keras import backend as K
from keras import activations
from keras.engine.topology import Layer, InputSpec
from keras.initializers import RandomNormal
from keras.layers import Add, Dense, Embedding, Input, Dropout
from keras.models import Model, load_model
import tensorflow as tf


# TODO
# - visualize attention
# - load method
# - enable encoder/decoder only architectures
# - docstrings/comments


logger = logging.getLogger(__name__)


class Transformer(Model):
    """Transformer model class

    Args:
        encoder_layers (int): The number of encoder layers.
        decoder_layers (int): The number of decoder layers.
        n_heads (int): The number of heads to use in multi-head attention
            layers.
        d_model (int): The dimension of the word embeddings used. Corresponds
            the value `d_model` in the paper. Must be a divisor of `n_heads`.
        d_k (int or None): If an `int` this is the dimension of the linear
            transformation applied to the "keys" in multi-head-attention
            layers. Corresponds to the value `d_k` in the paper. If `None`
            defaults to `d_model / n_heads`.
        d_v (int or None): If an `int` this is the dimension of the linear
            transformation applied to the "values" in multi-head-attention
            layers. Corresponds to the value `d_v` in the paper. If `None`
            defaults to `d_model / n_heads`.
        vocab_size (int): The number of unique tokens to create embeddings
            for.
        sequence_len (int or None): The length of the sequences. If an `int`
            is passed both the encoder/decoder embeddings will be initialized
            with this value as the sequence length. If `None` variable length
            sequences are allowed for both the encoder and decoder. `None` by
            default.
        preprocess_steps (list): `list` of `str` or `tuple` values describing
            how sublayers are to be preprocessed. Valid values are
            `('dropout', <dropout rate>)` and `'layer_normalization'`. Steps
            will be applied to the layer in the order they appear. (Optional)
        postprocess_steps (list): Same as `preprocess_steps`
            with the addition of being able to pass `'residual_connections'`
            in the list. (Optional)
        share_embedding_weights (bool): Whether the embedding weights should
            be shared with the final linear transformation in the model.
        positional_encoding_dropout (bool or float): If `False` dropout is not
            applied to the word embeddings after positional encoding is
            applied. If a `float`, dropout is applied and
            `positional_encoding_dropout` is the rate used.
        output_activation (str): Any valid keras activation argument. Defaults
            to `'softmax'`
        inputs (None): This is a `keras` load/save implementation detail. Do
            not pass.
        outputs (None): This is a `keras` load/save implementation detail. Do
            not pass.
        **kwargs: This is a `keras` load/save implementation detail. Do
            not pass.
    """
    def __init__(self, *,
                 encoder_layers=None, decoder_layers=None,
                 n_heads=None, d_model=None, d_k=None, d_v=None,
                 vocab_size=None, sequence_len=None,
                 preprocess_steps=None, postprocess_steps=None,
                 share_embedding_weights=True,
                 positional_encoding_dropout=0.1,
                 output_activation='softmax',
                 inputs=None, outputs=None, **kwargs):
        # define attributes
        self.n_heads = n_heads
        self.sequence_len = sequence_len
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.positional_encoding_dropout = positional_encoding_dropout
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.vocab_size = vocab_size        
        self.preprocess_steps = [] if preprocess_steps is None else preprocess_steps
        self.postprocess_steps = [] if postprocess_steps is None else postprocess_steps
        self.share_embedding_weights = share_embedding_weights
        self.output_activation = output_activation

        # set inputs/outputs appropriate 
        if inputs is None and outputs is None:
            inputs, outputs = self.build_inputs_outputs()
        else:
            self.decoder = outputs
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def build_inputs_outputs(self):
        self.encoder_input_spec, self.decoder_input_spec = self.init_input()
        self.encoder_layer_input, self.decoder_layer_input, self.embedding_weights = self.init_embeddings()
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        return [self.encoder_input_spec, self.decoder_input_spec], self.decoder

    def init_input(self):
        encoder_input_spec = Input(shape=(None,), name='encoder_input_spec')
        decoder_input_spec = Input(shape=(None,), name='decoder_input_spec')
        return encoder_input_spec, decoder_input_spec

    def init_embeddings(self):
        embedding = Embedding(input_dim=self.vocab_size, output_dim=self.d_model,
                              input_length=self.sequence_len, name='embedding')
        embedding_scalar = Scalar(np.sqrt(self.d_model), name='embedding_scalar')
        positional_encoding = PositionalEncoding()

        dropout = self.positional_encoding_dropout  # alias

        encoder_layer_input = embedding(self.encoder_input_spec)
        encoder_layer_input = positional_encoding(encoder_layer_input)
        encoder_layer_input = embedding_scalar(encoder_layer_input)
        if dropout:
            encoder_layer_input = Dropout(dropout)(encoder_layer_input)

        decoder_layer_input = embedding(self.decoder_input_spec)
        decoder_layer_input = positional_encoding(decoder_layer_input)
        decoder_layer_input = embedding_scalar(decoder_layer_input)
        if dropout:
            decoder_layer_input = Dropout(dropout)(decoder_layer_input)

        embedding_weights = embedding.embeddings if self.share_embedding_weights else None

        return encoder_layer_input, decoder_layer_input, embedding_weights

    def init_encoder(self):
        logger.debug('building encoder')
        encoder_layer_input = self.encoder_layer_input
        for _ in range(self.encoder_layers):
            # self attention
            encoder_layer_input = self.apply_sublayer_processing(
                encoder_layer_input, None, self.preprocess_steps)
            encoder_sublayer1 = MultiHeadAttention(n_heads=self.n_heads,
                                                   d_model=self.d_model,
                                                   d_k=self.d_k,
                                                   d_v=self.d_v)(encoder_layer_input)
            encoder_sublayer1 = self.apply_sublayer_processing(
                encoder_sublayer1, encoder_layer_input, self.postprocess_steps)
            # ffn
            encoder_sublayer1 = self.apply_sublayer_processing(
                encoder_sublayer1, None, self.preprocess_steps)
            encoder_sublayer2 = Dense(self.d_model, activation='relu')(encoder_sublayer1)
            encoder_sublayer2 = Dense(self.d_model)(encoder_sublayer2)
            encoder_sublayer2 = self.apply_sublayer_processing(
                encoder_sublayer2, encoder_sublayer1, self.postprocess_steps)
            # output of layer is input of next layer
            encoder_layer_input = encoder_sublayer2
        return encoder_sublayer2

    def init_decoder(self):
        logger.debug('building decoder')
        decoder_layer_input = self.decoder_layer_input
        for _ in range(self.decoder_layers):
            # self attention
            decoder_layer_input = self.apply_sublayer_processing(
                decoder_layer_input, None, self.preprocess_steps)
            decoder_sublayer1 = MultiHeadAttention(n_heads=self.n_heads,
                                                   d_model=self.d_model,
                                                   d_k=self.d_k, d_v=self.d_v,
                                                   masking=True)(decoder_layer_input)
            decoder_sublayer1 = self.apply_sublayer_processing(
                decoder_sublayer1, decoder_layer_input, self.postprocess_steps)
            # attention with encoder
            decoder_sublayer1 = self.apply_sublayer_processing(
                decoder_sublayer1, None, self.preprocess_steps)
            decoder_sublayer2 = MultiHeadAttention(
                n_heads=self.n_heads, d_model=self.d_model, d_k=self.d_k,
                d_v=self.d_v)([decoder_sublayer1,self.encoder,self.encoder])
            decoder_sublayer2 = self.apply_sublayer_processing(
                decoder_sublayer2, decoder_sublayer1, self.postprocess_steps)
            # ffn
            decoder_sublayer2 = self.apply_sublayer_processing(
                decoder_sublayer2, None, self.preprocess_steps)
            decoder_sublayer3 = Dense(self.d_model, activation='relu')(decoder_sublayer2)
            decoder_sublayer3 = Dense(self.d_model)(decoder_sublayer3)
            decoder_sublayer3 = self.apply_sublayer_processing(
                decoder_sublayer3, decoder_sublayer2, self.postprocess_steps)
            # output of layer is input of next layer
            decoder_layer_input = decoder_sublayer3
        # finally stack a linear transformation with softmax activation to get
        # token probabilities
        if self.share_embedding_weights:
            embedding_weights_T = K.transpose(self.embedding_weights)
            transform = SharedWeights(embedding_weights_T, activation=self.output_activation)
        else:
            transform = Dense(self.vocab_size, activation=self.output_activation)
        decoder = transform(decoder_sublayer3)
        return decoder

    @staticmethod
    def apply_sublayer_processing(sublayer1, sublayer2, processing_steps):
        """Apply `processing_steps` to `sublayer1`.

        Args:
            sublayer1 (keras tensor): layer to apply `processing_steps` to.
            sublayer2 (keras tensor): Only used if `'residual_connections'` is
                in `processing_steps`.
            processing_steps (`list`): A list as described by the parameters
                `preprocess_steps` and `postprocess_steps` of the
                `Transformer` class.
        """
        for step in processing_steps:
            if not isinstance(step, str):
                step, *args = step
            if step.lower() == 'dropout':
                rate = args[0] if isinstance(args[0], float) else 0.1
                sublayer1 = Dropout(rate)(sublayer1)
            elif step.lower() == 'layer_normalization':
                sublayer1 = LayerNormalization()(sublayer1)
            elif step.lower() == 'residual_connections':
                assert sublayer1 is not None, \
                    'cannot apply residual connections on input'
                sublayer1 = Add()([sublayer1, sublayer2])
        return sublayer1

    def get_config(self):
        config = super().get_config()
        config['n_heads'] = self.n_heads
        config['sequence_len'] = self.sequence_len
        config['encoder_layers'] = self.encoder_layers
        config['decoder_layers'] = self.decoder_layers
        config['positional_encoding_dropout'] = self.positional_encoding_dropout
        config['d_model'] = self.d_model
        config['d_k'] = self.d_k
        config['d_v'] = self.d_v
        config['vocab_size'] = self.vocab_size
        config['preprocess_steps'] = self.preprocess_steps
        config['postprocess_steps'] = self.postprocess_steps
        config['share_embedding_weights'] = self.share_embedding_weights
        config['output_activation'] = self.output_activation
        return config


class MultiHeadAttention(Layer):
    def __init__(self, n_heads, d_model, d_k=None, d_v=None, masking=False, **kwargs):
        # activation = comparison
        logger.debug('init MultiHeadAttention')
        assert d_model % n_heads == 0, 'h must divide d_model evenly'
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_model // n_heads if d_k is None else d_k
        self.d_v = self.d_model // n_heads if d_v is None else d_v
        self.masking = masking
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        if isinstance(input_shape, list):
            shape = input_shape[0]
        else:
            shape = input_shape
        logger.debug('building MultiAttention')
        self.W_o = self.add_weight(name='W_o', 
                                   shape=(self.n_heads*self.d_v, self.d_model),
                                   initializer='uniform',
                                   trainable=True)
        self.heads = [AttentionHead(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, activation='softmax')
                      for _ in range(self.n_heads)]
        super().build(input_shape)
    
    # this signature is a hack to work with keras layers call only adding
    # a single position tensor to the graph (causes problems in encoder-decoder
    # attention)
    def call(self, inputs):
        concat = K.concatenate([head(inputs, masking=self.masking) for head in self.heads])
        logger.debug('concat shape: %s', K.int_shape(concat))
        return K.dot(concat, self.W_o)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            shape = input_shape[0]
        else:
            shape = input_shape
        return (shape[0], shape[1], self.d_model)

    def get_config(self):
        config = super().get_config()
        config['n_heads'] = self.n_heads
        config['d_model'] = self.d_model
        config['masking'] = self.masking
        return config


class AttentionHead(Layer):
    def __init__(self, d_model, d_k, d_v, activation, **kwargs):
        logger.debug('init AttentionHead') 
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.scalar = np.sqrt(self.d_k)
        self._activation = activation
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            shape = input_shape[0]
        else:
            shape = input_shape
        self.W_q = self.add_weight(name='W_q',
                                   shape=(shape[-1], self.d_k),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        self.W_k = self.add_weight(name='W_k',
                                   shape=(shape[-1], self.d_k),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        self.W_v = self.add_weight(name='W_v',
                                   shape=(shape[-1], self.d_v),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        super().build(input_shape)

    def call(self, inputs, masking=False):
        try:
            q, k, v = inputs
        except TypeError:
            q = k = v = inputs
        q_p = K.dot(q, self.W_q)
        k_p = K.dot(k, self.W_k)
        v_p = K.dot(v, self.W_v)
        k_t = K.permute_dimensions(K.transpose(k_p), (2, 0, 1))
        attention_weights = K.batch_dot(q_p, k_t) / K.variable(self.scalar)
        if masking:
            logger.debug('masking')
            attention_weights = self.mask(attention_weights)
        x = self.activation(attention_weights)
        return K.batch_dot(x, v_p)

    @staticmethod
    def mask(x):
        shape = K.shape(x)
        mask = K.zeros((shape[1], shape[2])) + (-1e15)
        mask = tf.matrix_band_part(mask, 0, -1)  # upper triangle of `mask`
        mask -= tf.matrix_band_part(mask, 0, 0)  # remove diagonal
        return x + mask

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            assert len(set(input_shape)) == 1, 'k, q, and v must be of same shape'
            shape = input_shape[0]
        else:
            shape = input_shape
        return (shape[-1], self.d_v)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['d_k'] = self.d_k
        config['d_v'] = self.d_v
        config['activation'] = self._activation
        return config


class PositionalEncoding(Layer):
    def call(self, inputs):
        sequence_dim = K.shape(inputs)[1]
        d_model_var = K.shape(inputs)[2]
        d_model_int = K.int_shape(inputs)[2]
        rows, cols = self.indices(sequence_dim, d_model_var)
        rows, cols = K.cast(rows, dtype=K.floatx()), K.cast(cols, dtype=K.floatx())
        numerator = K.switch(cols % 2, K.cos(rows), K.sin(rows))
        denominator = 10_000**((2*cols)/d_model_int)
        return inputs + (numerator / denominator)

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def indices(dim1, dim2):
        """Return array representing the indices of a grid. 

        Like `numpy.indices` but works with `keras` types
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.indices.html
        """
        rows = K.arange(dim1)
        cols = K.arange(dim2)
        col_indices = K.reshape(K.tile(cols, [dim1]), (dim1, dim2))
        row_indices = K.transpose(K.reshape(K.tile(rows, [dim2]), (dim2, dim1)))
        return row_indices, col_indices


class Scalar(Layer):
    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    def call(self, x):
        return x * self.value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['value'] = self.value
        return config


class LayerNormalization(Layer):
    def __init__(self, **kwargs):
        self.epsilon = 1e-6
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gain = self.add_weight(name='gain',
                                    shape=(input_shape[-1],),
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[-1],),
                                    initializer='glorot_uniform',
                                    trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True) + self.epsilon
        x = ((inputs - mean) / std)
        return self.gain * x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class SharedWeights(Layer):
    def __init__(self, shared_weights, activation, **kwargs):
        self.x = shared_weights
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        out = K.dot(inputs, self.x)
        return self.activation(out)

    def compute_output_shape(self, input_shape):
        out = K.int_shape(self.x)[-1]
        return (input_shape[0], input_shape[1], out)


def init_cli():
    parser = argparse.ArgumentParser('debug interface to attention is all you need model')
    parser.add_argument('--summarize-model', action='store_true', default=False)
    parser.add_argument('--summarize-encoder', action='store_true', default=False)
    parser.add_argument('--plot-model', action='store_true', default=False)
    parser.add_argument('--plot-encoder', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    cli = parser.parse_args(sys.argv[1:])
    return cli


CUSTOM_OBJECTS = {
    'Transformer': Transformer,
    'MultiHeadAttention': MultiHeadAttention,
    'AttentionHead': AttentionHead,
    'PositionalEncoding': PositionalEncoding,
    'Scalar': Scalar,
    'LayerNormalization': LayerNormalization,
    'SharedWeights': SharedWeights,
}

if __name__ == '__main__':
    import sys

    CLI = init_cli()
    _ = logging.basicConfig(level='DEBUG') if CLI.debug  else None

    model = Transformer(
        encoder_layers=6,
        decoder_layers=6,
        n_heads=8,
        d_model=512,
        d_k=None,
        d_v=None,
        vocab_size=10_000,
        sequence_len=None,
        preprocess_steps=['layer_normalization'],
        postprocess_steps=[('dropout', 0.1), 'residual_connections'],
        share_embedding_weights=True,
        positional_encoding_dropout=0.1,
        output_activation='softmax')

    if CLI.summarize_model:
        print('MODEL SUMMARY')
        model.summary(line_length=100)
    if CLI.plot_model:
        keras.utils.plot_model(model, 'model.png', show_shapes=True)
        sp.call(['open', 'model.png'])
    if CLI.save_model:
        x = np.random.randint(0, VOCAB_SIZE, size=SEQUENCE_LEN).reshape(1, -1)
        X = [x, x[:,:-1]]
        y = np.eye(VOCAB_SIZE)[x[:,1:]]
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(X, y, epochs=40)
        p1 = model.predict(X)
        model.save('test_model_save.h5')
        model = load_model('test_model_save.h5', custom_objects=CUSTOM_OBJECTS)
        p2 = model.predict(X)
        print(p1, p2)
        assert np.allclose(p1, p2), 'weights not saved and/or loaded properly'

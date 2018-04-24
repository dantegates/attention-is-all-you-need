"""
Implementation of Transformer model, as described here

    https://arxiv.org/pdf/1706.03762.pdf
"""

from __future__ import absolute_import, division, print_function

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


# TODO
# - keyword only arguments
# - visualize attention
# - load method


logger = logging.getLogger(__name__)


class Transformer(Model):
    def __init__(self, n_heads=None, sequence_len=None, encoder_layers=None,
                 decoder_layers=None, d_model=None, d_k=None, d_v=None, vocab_size=None,
                 layer_normalization=True, dropout=0.1, residual_connections=True,
                 share_embedding_weights=True, output_activation='softmax'):
        # define attributes
        self.n_heads = n_heads
        self.sequence_len = sequence_len
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.vocab_size = vocab_size
        self.layer_normalization = layer_normalization
        self.dropout = dropout
        self.residual_connections = residual_connections
        self.share_embedding_weights = share_embedding_weights
        self.output_activation = output_activation

        self.encoder_input, self.decoder_input = self.init_input()
        self.encoder_embedding, self.decoder_embedding, self.embedding_weights = self.init_embeddings()
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        self.encoder_model = Model(self.encoder_input, self.encoder)
        super().__init__(inputs=[self.encoder_input, self.decoder_input],
                         outputs=self.decoder)

    def init_input(self):
        encoder_input = Input(shape=(None,), name='encoder_input')
        decoder_input = Input(shape=(None,), name='decoder_input')
        return encoder_input, decoder_input

    def init_embeddings(self):
        embedding = Embedding(input_dim=self.vocab_size, output_dim=self.d_model,
                              input_length=self.sequence_len, name='embedding')
        embedding_scalar = Scalar(np.sqrt(self.d_model), name='embedding_scalar')
        positional_encoding = PositionalEncoding(self.d_model, self.sequence_len)        

        encoder_embedding = embedding(self.encoder_input)
        encoder_embedding = positional_encoding(encoder_embedding)
        encoder_embedding = embedding_scalar(encoder_embedding)
        if self.dropout:
            encoder_embedding = Dropout(0.1)(encoder_embedding)

        decoder_embedding = embedding(self.decoder_input)
        decoder_embedding = positional_encoding(decoder_embedding)
        decoder_embedding = embedding_scalar(decoder_embedding)
        if self.dropout:
            decoder_embedding = Dropout(self.dropout)(decoder_embedding)

        embedding_weights = embedding.embeddings if self.share_embedding_weights else None

        return encoder_embedding, decoder_embedding, embedding_weights

    def init_encoder(self):
        # make encoder
        logger.debug('making encoder')
        encoder_layer_input = self.encoder_embedding
        for i in range(1, self.encoder_layers+1):
            names = iter([
                'encoder_layer%s_mha' % i,
                'encoder_layer%s_residual1' % i,
                'encoder_layer%s_layernorm1' % i,
                'encoder_layer%s_ffn1' % i,
                'encoder_layer%s_ffn2' % i,
                'encoder_layer%s_residual2' % i,
                'encoder_layer%s_layernorm2' % i,
            ])
            encoder_sublayer1 = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model,
                                                   d_k=self.d_k, d_v=self.d_v, name=next(names))
            encoder_sublayer1 = encoder_sublayer1(encoder_layer_input)
            if self.dropout:
                encoder_sublayer1 = Dropout(self.dropout)(encoder_sublayer1)
            if self.residual_connections:
                encoder_sublayer1 = Add(name=next(names))([encoder_layer_input, encoder_sublayer1])
            if self.layer_normalization:
                encoder_sublayer1 = LayerNormalization(name=next(names))(encoder_sublayer1)
            encoder_sublayer2 = Dense(self.d_model, activation='relu', name=next(names))(encoder_sublayer1)
            encoder_sublayer2 = Dense(self.d_model, name=next(names))(encoder_sublayer2)
            if self.dropout:
                encoder_sublayer2 = Dropout(self.dropout)(encoder_sublayer2)
            if self.residual_connections:
                encoder_sublayer2 = Add(name=next(names))([encoder_sublayer1, encoder_sublayer2])
            if self.layer_normalization:
                encoder_sublayer2 = LayerNormalization(name=next(names))(encoder_sublayer2)
            encoder_layer_input = encoder_sublayer2
        # finally pull it all together in a model
        return encoder_sublayer2

    def init_decoder(self):
        # make decoder
        decoder_layer_input = self.decoder_embedding
        logger.debug('making decoder')
        for i in range(1, self.decoder_layers+1):
            names = iter([
                'decoder_layer%s_mha1' % i,
                'decoder_layer%s_residual1' % i,
                'decoder_layer%s_layernorm1' % i,
                'decoder_layer%s_mha2' % i,
                'decoder_layer%s_residual2' % i,
                'decoder_layer%s_layernorm2' % i,
                'decoder_layer%s_ffn1' % i,
                'decoder_layer%s_ffn2' % i,
                'decoder_layer%s_residual3' % i,
                'decoder_layer%s_layernorm3' % i,
            ])
            decoder_sublayer1 = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model,
                                                   d_k=self.d_k, d_v=self.d_v,
                                                   masking=True, name=next(names))
            decoder_sublayer1 = decoder_sublayer1(decoder_layer_input)
            if self.dropout:
                decoder_sublayer1 = Dropout(self.dropout)(decoder_sublayer1)
            if self.residual_connections:
                decoder_sublayer1 = Add(name=next(names))([decoder_layer_input, decoder_sublayer1])
            if self.layer_normalization:
                decoder_sublayer1 = LayerNormalization(name=next(names))(decoder_sublayer1)
            decoder_sublayer2 = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model,
                                                   d_k=self.d_k, d_v=self.d_v, name=next(names))
            decoder_sublayer2 = decoder_sublayer2([decoder_sublayer1, self.encoder, self.encoder])
            if self.dropout:
                decoder_sublayer2 = Dropout(self.dropout)(decoder_sublayer2)
            if self.residual_connections:
                decoder_sublayer2 = Add(name=next(names))([decoder_sublayer1, decoder_sublayer2])
            if self.layer_normalization:
                decoder_sublayer2 = LayerNormalization(name=next(names))(decoder_sublayer2)
            decoder_sublayer3 = Dense(self.d_model, activation='relu', name=next(names))(decoder_sublayer2)
            decoder_sublayer3 = Dense(self.d_model, name=next(names))(decoder_sublayer3)
            if self.dropout:
                decoder_sublayer3 = Dropout(self.dropout)(decoder_sublayer3)
            if self.residual_connections:
                decoder_sublayer3 = Add(name=next(names))([decoder_sublayer2, decoder_sublayer3])
            if self.layer_normalization:
                decoder_sublayer3 = LayerNormalization(name=next(names))(decoder_sublayer3)
            # output of layer becomes input of next layer
            decoder_layer_input = decoder_sublayer3
        # finally stack a linear transformation with softmax activation
        # to get token probabilities
        if self.share_embedding_weights:
            final_output = SharedWeights(K.transpose(self.embedding_weights), activation=self.output_activation)
        else:
            final_output = Dense(self.vocab_size, activation=self.output_activation)
        decoder = final_output(decoder_sublayer3)
        return decoder

    def get_config(self):
        config = super().get_config()
        config['n_heads'] = self.n_heads
        config['sequence_len'] = self.sequence_len
        config['encoder_layers'] = self.encoder_layers
        config['decoder_layers'] = self.decoder_layers
        config['d_model'] = self.d_model
        config['d_k'] = self.d_k
        config['d_v'] = self.d_v
        config['vocab_size'] = self.vocab_size
        config['layer_normalization'] = self.layer_normalization
        config['dropout'] = self.dropout
        config['residual_connections'] = self.residual_connections
        return config


class MultiHeadAttention(Layer):
    def __init__(self, n_heads, d_model, d_k, d_v, masking=False, **kwargs):
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
            assert len(set(input_shape)) == 1, 'k, q, and v must be of same shape'
            shape = input_shape[0]
        else:
            shape = input_shape
        logger.debug('building MultiAttention')
        self.W_o = self.add_weight(name='W_o', 
                                   shape=(self.n_heads*self.d_v, self.d_model),
                                   initializer='uniform',
                                   trainable=True)
        self.heads = [Attention(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, activation='softmax')
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
            assert len(set(input_shape)) == 1, 'k, q, and v must be of same shape'
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


class Attention(Layer):
    def __init__(self, d_model, d_k, d_v, activation, **kwargs):
        logger.debug('init Attention') 
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.scalar = np.sqrt(self.d_k)
        self._activation = activation
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(set(input_shape)) == 1, 'k, q, and v must be of same shape'
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

    def mask(self, x):
        shape = K.int_shape(x)
        assert shape[1] == shape[2], 'expected square matrix'
        mask = np.zeros((shape[1], shape[1]))
        invalid_indices = np.triu_indices(shape[1], 1)
        mask[invalid_indices] = 1e-15
        mask = K.variable(mask)
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
    def __init__(self, d_model, sequence_len, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.sequence_len = sequence_len
        self.encoding = self.make_encoding(d_model, sequence_len)

    def call(self, inputs):
        return self.encoding + inputs

    def make_encoding(self, d_model, sequence_len):
        def gen():
            for i in range(d_model):
                f = np.sin if i % 2 == 0 else np.cos
                yield f(np.arange(sequence_len) / ((10000**(2*i/d_model))))
        arr = np.array(list(gen())).transpose()
        return K.variable(arr)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['sequence_len'] = self.sequence_len
        return config


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


class FFN(Layer):
    def __init__(self, units, activation, **kwargs):
        self.units = units
        self._activation = activation
        self.activation = activations.get(activation)
        self.bias_regularizer = None
        self.bias_constraint = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel1 = self.add_weight(shape=(input_dim, self.units),
                                       initializer='glorot_uniform',
                                       name='kernel1',
                                       regularizer=None,
                                       constraint=None,
                                       trainable=True)
        self.bias1 = self.add_weight(shape=(self.units,),
                                     initializer='zeros',
                                     name='bias1',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint,
                                     trainable=True)
        self.kernel2 = self.add_weight(shape=(input_dim, self.units),
                                       initializer='glorot_uniform',
                                       name='kernel2',
                                       regularizer=None,
                                       constraint=None,
                                       trainable=True)
        self.bias2 = self.add_weight(shape=(self.units,),
                                     initializer='zeros',
                                     name='bias2',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint,
                                     trainable=True)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        super().build(input_shape)

    def call(self, x):
        output = self.activation(K.bias_add(K.dot(x, self.kernel1), self.bias1))
        return K.bias_add(K.dot(output, self.kernel2), self.bias2)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        config['activation'] = self._activation


class LayerNormalization(Layer):
    def __init__(self, **kwargs):
        self.epsilon = 1e-6
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gain = self.add_weight(name='gain',
                                    shape=input_shape[1:],
                                    initializer='glorot_uniform',
                                    trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=input_shape[1:],
                                    initializer='glorot_uniform',
                                    trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return (self.gain / (std + self.epsilon)) * (x - mean) + self.bias

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


if __name__ == '__main__':
    import sys

    N_HEADS = 8
    ENCODER_LAYERS = DECODER_LAYERS = 2
    D_MODEL = 64 * N_HEADS
    VOCAB_SIZE = 32
    SEQUENCE_LEN = 30
    SHARE_EMBEDDING_WEIGHTS = False
    CLI = init_cli()
    _ = logging.basicConfig(level='DEBUG') if CLI.debug  else None

    model = Transformer(
        n_heads=N_HEADS, encoder_layers=ENCODER_LAYERS, decoder_layers=DECODER_LAYERS,
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, sequence_len=SEQUENCE_LEN,
        share_embedding_weights=SHARE_EMBEDDING_WEIGHTS)

    if CLI.summarize_encoder:
        print('ENCODER SUMMARY')
        model.encoder_model.summary(line_length=100)
    if CLI.summarize_model:
        print('MODEL SUMMARY')
        model.summary(line_length=100)
    if CLI.plot_encoder:
        keras.utils.plot_model(model.encoder_model, 'encoder.png', show_shapes=True)
        sp.call(['open', 'encoder.png'])
    if CLI.plot_model:
        keras.utils.plot_model(model, 'model.png', show_shapes=True)
        sp.call(['open', 'model.png'])
    if CLI.save_model:
        X = np.random.randint(0, VOCAB_SIZE, size=SEQUENCE_LEN)
        X = [X, X]
        p1 = model.predict(X)
        model.save('test_model_save.h5')
        # so far this is an unsatisfying, solution.
        # model = Transformer(
        #     n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        #     d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len)
        model = load_model('test_model_save.h5', custom_objects=locals())
        p2 = model.predict(X)
        print(p1, p2)
        assert (p1 == p2).all(), 'weights not saved and/or loaded properly'

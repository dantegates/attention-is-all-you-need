"""
Modified version of Transformer model, including only the decoder
as described here

    https://arxiv.org/pdf/1801.10198.pdf
    
TODO: Both T-ED and T-D should be available from the same class in
    model.py. Keeping them separate for now though while still
    experimenting with code.
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


class TransformerDecoder(Model):
    def __init__(self, n_heads=None, sequence_len=None,
                 decoder_layers=None, d_model=None, vocab_size=None,
                 dropout=0.1, sparse=False):
        # define attributes
        self.n_heads = n_heads
        self.sequence_len = sequence_len
        self.decoder_layers = decoder_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.sparse = sparse

        self.decoder_input = self.init_input()
        self.decoder_embedding, self.embedding_weights = self.init_embeddings()
        self.decoder = self.init_decoder()
        super().__init__(inputs=self.decoder_input, outputs=self.decoder)

    def init_input(self):
        return Input(shape=(None,), name='input')

    def init_embeddings(self):
        embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.d_model,
                              input_length=self.sequence_len, name='embedding')
        embedding_scalar = Scalar(np.sqrt(self.d_model), name='embedding_scalar')
        positional_encoding = PositionalEncoding(self.d_model, self.sequence_len)

        embedding = embedding_layer(self.decoder_input)
        embedding = positional_encoding(embedding)
        embedding = embedding_scalar(embedding)
        if self.dropout:
            embedding = Dropout(self.dropout)(embedding)

        return embedding, embedding_layer.embeddings

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
            # Sublayer 1
            decoder_sublayer1 = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model,
                                                   masking=True, name=next(names))
            decoder_sublayer1 = decoder_sublayer1(decoder_layer_input)
            if self.dropout:
                decoder_sublayer1 = Dropout(self.dropout)(decoder_sublayer1)
            decoder_sublayer1 = Add(name=next(names))([decoder_layer_input, decoder_sublayer1])
            decoder_sublayer1 = LayerNormalization(name=next(names))(decoder_sublayer1)

            # Sublayer 2
            decoder_sublayer2 = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model, name=next(names))
            decoder_sublayer2 = decoder_sublayer2(decoder_sublayer1)
            if self.dropout:
                decoder_sublayer2 = Dropout(self.dropout)(decoder_sublayer2)
            decoder_sublayer2 = Add(name=next(names))([decoder_sublayer1, decoder_sublayer2])
            decoder_sublayer2 = LayerNormalization(name=next(names))(decoder_sublayer2)

            # Sublayer 3
            decoder_sublayer3 = Dense(self.d_model, activation='relu', name=next(names))(decoder_sublayer2)
            decoder_sublayer3 = Dense(self.d_model, name=next(names))(decoder_sublayer3)
            if self.dropout:
                decoder_sublayer3 = Dropout(self.dropout)(decoder_sublayer3)
            decoder_sublayer3 = Add(name=next(names))([decoder_sublayer2, decoder_sublayer3])
            decoder_sublayer3 = LayerNormalization(name=next(names))(decoder_sublayer3)
            # output of layer becomes input of next layer
            decoder_layer_input = decoder_sublayer3
        # finally stack a linear transformation with softmax activation
        # to get token probabilities
        #
        # linear activation is a hack while keras sparse_categorical_crossentropy does not
        # seem to work.
        # see: https://github.com/tensorflow/tensorflow/issues/17150
        final_activation = 'linear' if self.sparse else 'softmax'
        final_output = SharedWeights(K.transpose(self.embedding_weights), activation=final_activation)
        decoder = final_output(decoder_sublayer3)
        return decoder

    def get_config(self):
        config = super().get_config()
        config['n_heads'] = self.n_heads
        config['sequence_len'] = self.sequence_len
        config['decoder_layers'] = self.decoder_layers
        config['d_model'] = self.d_model
        config['vocab_size'] = self.vocab_size
        config['layer_normalization'] = self.layer_normalization
        config['dropout'] = self.dropout
        config['residual_connections'] = self.residual_connections
        return config


class MultiHeadAttention(Layer):
    def __init__(self, n_heads, d_model, masking=False, **kwargs):
        # activation = comparison
        logger.debug('init MultiHeadAttention')
        self.n_heads = n_heads
        self.d_model = d_model
        assert self.d_model % n_heads == 0, 'h must divide d_model evenly'
        self.d_k = self.d_v = self.d_model // n_heads
        self.masking = masking
        super().__init__(**kwargs)

    def build(self, input_shape):
        shape = input_shape
        logger.debug('building MultiAttention')
        self.W_o = self.add_weight(name='W_o', 
                                   shape=(self.n_heads*self.d_v, self.d_model),
                                   initializer='uniform',
                                   trainable=True)
        self.heads = [AttentionHead(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, activation='softmax')
                      for _ in range(self.n_heads)]
        super().build(input_shape)
    
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


class AttentionHead(Layer):
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

    n_heads = 8
    decoder_layers = 2
    d_model = 64 * n_heads
    vocab_size = 32
    sequence_len = 30
    test_sequence_len = 100
    cli = init_cli()
    _ = logging.basicConfig(level='DEBUG') if cli.debug else None

    model = TransformerDecoder(
        n_heads=n_heads, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len)

    if cli.summarize_encoder:
        print('ENCODER SUMMARY')
        model.encoder_model.summary(line_length=100)
    if cli.summarize_model:
        print('MODEL SUMMARY')
        model.summary(line_length=100)
    if cli.plot_encoder:
        keras.utils.plot_model(model.encoder_model, 'encoder.dot', show_shapes=True)
        sp.call(['dot', '-Tpng', 'encoder.dot', '-o', 'encoder.png'])
        sp.call(['open', 'encoder.png'])
    if cli.plot_model:
        keras.utils.plot_model(model, 'model.dot', show_shapes=True)
        sp.call(['dot', '-Tpng', 'model.dot', '-o', 'model.png'])
        sp.call(['open', 'model.png'])
    if cli.save_model:
        X = np.random.randint(0, vocab_size, size=sequence_len)
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

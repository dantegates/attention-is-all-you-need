"""
Implementation of Transformer model, as described here

    https://arxiv.org/pdf/1706.03762.pdf
"""

from __future__ import division, print_function, absolute_import

import argparse

import keras
import numpy as np
from keras import backend as K
from keras import activations
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras.layers import Dense, Embedding, Input, Add, Lambda
from keras.models import Model

from data import training_data


# TODO:
# - Mask decoder
# - rename d_model to output_shape
# - share embedding weights with final linear transformation
# - dropout


DEBUG = False
def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class MultiHeadAttention(Layer):
    def __init__(self, h, d_model, **kwargs):
        # activation = comparison
        debug('init MultiHeadAttention')
        self.h = h
        self.d_model = d_model
        assert self.d_model % h == 0, 'h must divide d_model evenly'
        self.d_k = self.d_v = int(self.d_model / h)
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        debug('building MultiAttentionHead')
        self.k = self.add_weight(name='k', 
                                 shape=(1, self.d_model),
                                 initializer='uniform',
                                 trainable=True)
        self.v = self.add_weight(name='v', 
                                 shape=(1, self.d_model),
                                 initializer='uniform',
                                 trainable=True)
        self.W_o = self.add_weight(name='W_o', 
                                   shape=(self.h*self.d_v, self.d_model),
                                   initializer='uniform',
                                   trainable=True)
        self.heads = [AttentionHead(self.d_model, self.d_k, self.d_v, activation='softmax')
                      for _ in range(self.h)]
        super().build(input_shape)
    
    def call(self, q):
        concat = K.concatenate([head(q, k=self.k, v=self.v) for head in self.heads])
        debug('concat shape', K.int_shape(concat))
        return K.dot(concat, self.W_o)


class AttentionHead(Layer):
    def __init__(self, d_model, d_k, d_v, activation, masking=False, **kwargs):
        debug('init AttentionHead') 
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.scalar = np.sqrt(self.d_k)
        self.activation = activations.get(activation)
        self.masking = masking
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(name='W_q',
                                   shape=(self.d_model, self.d_k),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        self.W_k = self.add_weight(name='W_k',
                                   shape=(self.d_model, self.d_k),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        self.W_v = self.add_weight(name='W_v',
                                   shape=(self.d_model, self.d_k),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        super().build(input_shape)

    def call(self, q, k, v):
        q_proj = K.dot(q, self.W_q)
        k_proj = K.dot(k, self.W_k)
        v_proj = K.dot(v, self.W_v)
        value_weights = \
            self.activation(K.dot(q_proj, K.transpose(k_proj)) / self.scalar)
        debug('value weights shape', K.int_shape(value_weights))
        return K.dot(value_weights, v_proj)


class PositionalEncoding(Layer):
    def __init__(self, d_model, sequence_len, **kwargs):
        super().__init__(**kwargs)
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


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gain = self.add_weight(name='gain',
                                    shape=input_shape[1:],
                                    initializer='ones',
                                    trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=input_shape[1:],
                                    initializer='ones',
                                    trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return (self.gain / (std + self.epsilon)) * (x - mean) + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


def init_model(n_heads, encoder_layers, decoder_layers, d_model, vocab_size, sequence_len):

    # create embedding and model inputs
    embedding = Embedding(input_dim=vocab_size, output_dim=d_model,
                          input_length=sequence_len, name='embedding')
    embedding_scalar = Lambda(lambda x: x*np.sqrt(d_model), name='embedding_scalar')
    positional_encoding = PositionalEncoding(d_model, sequence_len)

    encoder_input = Input(shape=(None,), name='encoder_input')
    encoder_embedding = embedding(encoder_input)
    encoder_embedding = positional_encoding(encoder_embedding)
    encoder_embedding = embedding_scalar(encoder_embedding)

    decoder_input = Input(shape=(None,), name='decoder_input')
    decoder_embedding = embedding(decoder_input)
    decoder_embedding = positional_encoding(decoder_embedding)
    decoder_embedding = embedding_scalar(decoder_embedding)

    # make encoder
    encoder_layer_input = encoder_embedding
    for i in range(1, encoder_layers+1):
        names = iter([
            'encoder_layer%s_mha' % i,
            'encoder_layer%s_residual1' % i,
            'encoder_layer%s_layernorm1' % i,
            'encoder_layer%s_ffn1' % i,
            'encoder_layer%s_ffn2' % i,
            'encoder_layer%s_residual2' % i,
            'encoder_layer%s_layernorm2' % i,
        ])
        encoder = MultiHeadAttention(h=n_heads, d_model=d_model, name=next(names))(encoder_layer_input)
        encoder_sublayer1 = Add(name=next(names))([encoder_layer_input, encoder])
        encoder_sublayer1 = LayerNormalization(name=next(names))(encoder_sublayer1)
        encoder_sublayer2 = Dense(d_model, activation='relu', name=next(names))(encoder_sublayer1)
        encoder_sublayer2 = Dense(d_model, name=next(names))(encoder_sublayer2)
        encoder_sublayer2 = Add(name=next(names))([encoder_sublayer1, encoder_sublayer2])
        encoder_sublayer2 = LayerNormalization(name=next(names))(encoder_sublayer2)
        encoder_layer_input = encoder_sublayer2
    # finally pull it all together in a model
    encoder_model = Model(inputs=encoder_input, outputs=encoder_sublayer2)

    # make decoder
    decoder_layer_input = decoder_embedding
    for i in range(1, decoder_layers+1):
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
        decoder_sublayer1 = MultiHeadAttention(h=n_heads, d_model=d_model, name=next(names))(decoder_layer_input)
        decoder_sublayer1 = Add(name=next(names))([decoder_layer_input, decoder_sublayer1])
        decoder_sublayer1 = LayerNormalization(name=next(names))(decoder_sublayer1)
        decoder_sublayer2 = MultiHeadAttention(h=n_heads, d_model=d_model, name=next(names))(encoder)
        decoder_sublayer2 = Add(name=next(names))([decoder_sublayer1, decoder_sublayer2])
        decoder_sublayer2 = LayerNormalization(name=next(names))(decoder_sublayer2)
        decoder_sublayer3 = Dense(d_model, activation='relu', name=next(names))(decoder_sublayer2)
        decoder_sublayer3 = Dense(d_model, name=next(names))(decoder_sublayer3)
        decoder_sublayer3 = Add(name=next(names))([decoder_sublayer2, decoder_sublayer3])
        decoder_sublayer3 = LayerNormalization(name=next(names))(decoder_sublayer3)
        # output of layer becomes input of next layer
        decoder_layer_input = decoder_sublayer3
    # finally stack a linear transformation with softmax activation
    # to get next token probabilities
    decoder = Dense(vocab_size, activation='softmax')(decoder_sublayer3)
    decoder_model = Model(inputs=[encoder_input, decoder_input], outputs=decoder)

    return encoder_model, decoder_model


def init_cli():
    parser = argparse.ArgumentParser('debug interface to attention is all you need model')
    parser.add_argument('--summarize-models', action='store_true', default=False)
    parser.add_argument('--summarize-encoder', action='store_true', default=False)
    parser.add_argument('--summarize-decoder', action='store_true', default=False)
    parser.add_argument('--plot-models', action='store_true', default=False)
    parser.add_argument('--plot-encoder', action='store_true', default=False)
    parser.add_argument('--plot-decoder', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    cli = parser.parse_args(sys.argv[1:])
    return cli


if __name__ == '__main__':
    import sys

    n_heads = 8
    encoder_layers = decoder_layers = 6
    d_model = 64 * n_heads
    vocab_size = 30
    sequence_len = 30
    test_sequence_len = 100
    cli = init_cli()
    DEBUG = cli.debug

    encoder, decoder = init_model(
        n_heads=n_heads, encoder_layers=encoder_layers,
        decoder_layers=decoder_layers, d_model=d_model, vocab_size=vocab_size,
        sequence_len=sequence_len)

    if cli.summarize_models or cli.summarize_encoder:
        print('ENCODER SUMMARY')
        encoder.summary(line_length=100)
    if cli.summarize_models or cli.summarize_decoder:
        print('DECODER SUMMARY')
        decoder.summary(line_length=100)
    if cli.plot_models or cli.plot_encoder:
        keras.utils.plot_model(encoder, 'encoder.dot')
    if cli.plot_models or cli.plot_encoder:
        keras.utils.plot_model(decoder, 'decoder.dot')

    if cli.train:
        x, x, y = training_data(sequence_len)
        decoder.compile(loss='categorical_crossentropy', optimizer='adam')
        decoder.fit([x, x], y, batch_size=30, epochs=1)


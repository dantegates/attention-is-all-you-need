"""
Implementation of Transformer model, as described here

    https://arxiv.org/pdf/1706.03762.pdf
"""

from __future__ import absolute_import, division, print_function

import argparse

import keras
import numpy as np
from data import training_data
from keras import backend as K
from keras import activations
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras.layers import Add, Dense, Embedding, Input, Lambda
from keras.layers.advanced_activations import Softmax
from keras.models import Model

# TODO:
# - Mask decoder
# - share embedding weights with final linear transformation
# - dropout
# - learning rate decay during train


DEBUG = False
def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class MultiHeadAttention(Layer):
    def __init__(self, n_heads, d_model, masking=False, **kwargs):
        # activation = comparison
        debug('init MultiHeadAttention')
        self.n_heads = n_heads
        self.d_model = d_model
        assert self.d_model % n_heads == 0, 'h must divide d_model evenly'
        self.d_k = self.d_v = self.d_model // n_heads
        self.masking = masking
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        debug('building MultiAttention')
        self.W_o = self.add_weight(name='W_o', 
                                   shape=(self.n_heads*self.d_v, self.d_model),
                                   initializer='uniform',
                                   trainable=True)
        self.heads = [Attention(self.d_model, self.d_k, self.d_v, activation='softmax')
                      for _ in range(self.n_heads)]
        super().build(input_shape)
    
    def call(self, q, k, v):
        concat = K.concatenate([head(q, k=k, v=v, masking=self.masking) for head in self.heads])
        debug('concat shape', K.int_shape(concat))
        return K.dot(concat, self.W_o)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)


class Attention(Layer):
    def __init__(self, d_model, d_k, d_v, activation, **kwargs):
        debug('init Attention') 
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.scalar = np.sqrt(self.d_k)
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(name='W_q',
                                   shape=(input_shape[-1], self.d_k),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        self.W_k = self.add_weight(name='W_k',
                                   shape=(input_shape[-1], self.d_k),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        self.W_v = self.add_weight(name='W_v',
                                   shape=(input_shape[-1], self.d_v),
                                   initializer=RandomNormal(mean=0.0, stddev=1.0),
                                   trainable=True)
        super().build(input_shape)

    def call(self, q, k, v, masking=False):
        q_p = K.dot(q, self.W_q)
        k_p = K.dot(k, self.W_k)
        k_v = K.dot(v, self.W_v)
        k_t = K.permute_dimensions(K.transpose(k_p), (2, 0, 1))
        weights = K.batch_dot(q_p, k_t) / self.scalar
        if masking:
            debug('masking')
            weights = self.mask(weights)
        return K.batch_dot(weights, k_v)

    def mask(self, x):
        shape = K.int_shape(x)
        assert shape[1] == shape[2], 'expected square matrix'
        mask = np.zeros((shape[1], shape[1]))
        invalid_indices = np.triu_indices(shape[1], 1)
        mask[invalid_indices] = -np.inf
        mask = K.variable(mask)
        return x + mask

    def compute_output_shape(self, input_shape):
        return (input_shape[-1], self.d_v)


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

    def compute_output_shape(self, input_shape):
        return input_shape


class FFN(Layer):
    def __init__(self, **kwargs):
        self.activation = activations.get('relu')
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


def init_model(n_heads, encoder_layers, decoder_layers, d_model, vocab_size,
               sequence_len, n_outputs=None):

    # create embedding and model inputs
    embedding = Embedding(input_dim=vocab_size, output_dim=d_model,
                          input_length=sequence_len, name='embedding')
    embedding_scalar = Lambda(lambda x: x*np.sqrt(d_model),
                              output_shape=lambda x: x,
                              name='embedding_scalar')
    positional_encoding = PositionalEncoding(d_model, sequence_len)

    encoder_input = Input(shape=(None,), name='encoder_input')
    encoder_embedding = embedding(encoder_input)
    encoder_embedding = positional_encoding(encoder_embedding)
    encoder_embedding = embedding_scalar(encoder_embedding)

    # shared_weights = embedding.embeddings
    # final_transformation = Lambda(lambda x: K.dot(K.transpose(shared_weights), x))

    decoder_input = Input(shape=(None,), name='decoder_input')
    decoder_embedding = embedding(decoder_input)
    decoder_embedding = positional_encoding(decoder_embedding)
    decoder_embedding = embedding_scalar(decoder_embedding)

    # make encoder
    debug('making encoder')
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
        encoder = MultiHeadAttention(n_heads=n_heads, d_model=d_model, name=next(names))
        encoder = encoder(encoder_layer_input, k=encoder_layer_input, v=encoder_layer_input)
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
    debug('making decoder')
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
        decoder_sublayer1 = MultiHeadAttention(n_heads=n_heads, d_model=d_model, masking=True, name=next(names))
        decoder_sublayer1 = decoder_sublayer1(decoder_layer_input, k=decoder_layer_input, v=decoder_layer_input)
        decoder_sublayer1 = Add(name=next(names))([decoder_layer_input, decoder_sublayer1])
        decoder_sublayer1 = LayerNormalization(name=next(names))(decoder_sublayer1)
        decoder_sublayer2 = MultiHeadAttention(n_heads=n_heads, d_model=d_model, name=next(names))
        decoder_sublayer2 = decoder_sublayer2(decoder_sublayer1, k=encoder, v=encoder)
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
    # decoder = final_transformation(decoder_sublayer3)
    # decoder = Softmax()(decoder)
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
    vocab_size = 27
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
        x, x, y, vocab_size = training_data(sequence_len)
        decoder.compile(loss='categorical_crossentropy', optimizer='adam')
        decoder.fit([x, x], y, batch_size=30, epochs=1)

"""
Implementation of Transformer model, as described here

    https://arxiv.org/pdf/1706.03762.pdf
"""

from __future__ import absolute_import, division, print_function

import argparse
import subprocess as sp

import keras
import numpy as np
from keras import backend as K
from keras import activations
from keras.engine.topology import Layer, InputSpec
from keras.initializers import RandomNormal
from keras.layers import Add, Dense, Embedding, Input, Dropout
from keras.layers.advanced_activations import Softmax
from keras.models import Model

# TODO:
# - share embedding weights with final linear transformation
# - learning rate decay during train
# - proper logging
# - keyword only arguments


DEBUG = False
def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class Transformer(Model):
    def __init__(self, n_heads, encoder_layers, decoder_layers, d_model,
                 vocab_size, sequence_len, n_outputs=None):
        self.n_heads = n_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.sequence_len = sequence_len
        self.encoder_input, self.decoder_input = self.init_input()
        self.encoder_embedding, self.decoder_embedding = self.init_embeddings()
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

        # shared_weights = embedding.embeddings
        # final_transformation = Lambda(lambda x: K.dot(K.transpose(shared_weights), x))

        decoder_embedding = embedding(self.decoder_input)
        decoder_embedding = positional_encoding(decoder_embedding)
        decoder_embedding = embedding_scalar(decoder_embedding)
        return encoder_embedding, decoder_embedding

    def init_encoder(self):
        # make encoder
        debug('making encoder')
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
            encoder_sublayer1 = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model, name=next(names))
            encoder_sublayer1 = encoder_sublayer1(encoder_layer_input)
            encoder_sublayer1 = Dropout(0.1)(encoder_sublayer1)
            encoder_sublayer1 = Add(name=next(names))([encoder_layer_input, encoder_sublayer1])
            encoder_sublayer1 = LayerNormalization(name=next(names))(encoder_sublayer1)
            encoder_sublayer2 = Dense(self.d_model, activation='relu', name=next(names))(encoder_sublayer1)
            encoder_sublayer2 = Dense(self.d_model, name=next(names))(encoder_sublayer2)
            encoder_sublayer2 = Dropout(0.1)(encoder_sublayer2)
            encoder_sublayer2 = Add(name=next(names))([encoder_sublayer1, encoder_sublayer2])
            encoder_sublayer2 = LayerNormalization(name=next(names))(encoder_sublayer2)
            encoder_layer_input = encoder_sublayer2
        # finally pull it all together in a model
        return encoder_sublayer2

    def init_decoder(self):
        # make decoder
        decoder_layer_input = self.decoder_embedding
        debug('making decoder')
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
                                                   masking=True, name=next(names))
            decoder_sublayer1 = decoder_sublayer1(decoder_layer_input)
            decoder_sublayer1 = Dropout(0.1)(decoder_sublayer1)
            decoder_sublayer1 = Add(name=next(names))([decoder_layer_input, decoder_sublayer1])
            decoder_sublayer1 = LayerNormalization(name=next(names))(decoder_sublayer1)
            decoder_sublayer2 = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model, name=next(names))
            decoder_sublayer2 = decoder_sublayer2(self.encoder, q=decoder_sublayer1, v=self.encoder)
            decoder_sublayer2 = Dropout(0.1)(decoder_sublayer2)
            decoder_sublayer2 = Add(name=next(names))([decoder_sublayer1, decoder_sublayer2])
            decoder_sublayer2 = LayerNormalization(name=next(names))(decoder_sublayer2)
            decoder_sublayer3 = Dense(self.d_model, activation='relu', name=next(names))(decoder_sublayer2)
            decoder_sublayer3 = Dense(self.d_model, name=next(names))(decoder_sublayer3)
            decoder_sublayer3 = Dropout(0.1)(decoder_sublayer3)
            decoder_sublayer3 = Add(name=next(names))([decoder_sublayer2, decoder_sublayer3])
            decoder_sublayer3 = LayerNormalization(name=next(names))(decoder_sublayer3)
            # output of layer becomes input of next layer
            decoder_layer_input = decoder_sublayer3
        # finally stack a linear transformation with softmax activation
        # to get next token probabilities
        # decoder = final_transformation(decoder_sublayer3)
        # decoder = Softmax()(decoder)
        decoder = Dense(self.vocab_size, activation='softmax', name='decoder_output')(decoder_sublayer3)
        return decoder


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
        self.heads = [Attention(d_model=self.d_model, d_k=self.d_k, d_v=self.d_v, activation='softmax')
                      for _ in range(self.n_heads)]
        super().build(input_shape)
    
    # this signature is a hack to work with keras layers call only adding
    # a single position tensor to the graph (causes problems in encoder-decoder
    # attention)
    def call(self, k, q=None, v=None):
        if q is None and v is None:
            q = v = k    
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
        weights = K.batch_dot(q_p, k_t) / K.variable(self.scalar)
        if masking:
            debug('masking')
            weights = self.mask(weights)
        return K.batch_dot(weights, k_v)

    def mask(self, x):
        shape = K.int_shape(x)
        assert shape[1] == shape[2], 'expected square matrix'
        mask = np.zeros((shape[1], shape[1]))
        invalid_indices = np.triu_indices(shape[1], 1)
        mask[invalid_indices] = 1e-15
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


class Scalar(Layer):
    def __init__(self, value, **kwargs):
        self.value = value
        super().__init__(**kwargs)

    def call(self, x):
        return x * self.value

    def compute_output_shape(self, input_shape):
        return input_shape


class FFN(Layer):
    def __init__(self, units, activation, **kwargs):
        self.units = units
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


def init_cli():
    parser = argparse.ArgumentParser('debug interface to attention is all you need model')
    parser.add_argument('--summarize-model', action='store_true', default=False)
    parser.add_argument('--summarize-encoder', action='store_true', default=False)
    parser.add_argument('--plot-model', action='store_true', default=False)
    parser.add_argument('--plot-encoder', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    cli = parser.parse_args(sys.argv[1:])
    return cli


if __name__ == '__main__':
    import sys

    n_heads = 8
    encoder_layers = decoder_layers = 2
    d_model = 64 * n_heads
    vocab_size = 32
    sequence_len = 30
    test_sequence_len = 100
    cli = init_cli()
    DEBUG = cli.debug

    model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers,
        decoder_layers=decoder_layers, d_model=d_model, vocab_size=vocab_size,
        sequence_len=sequence_len)

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

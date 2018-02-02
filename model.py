"""
Implementation of Transformer model, as described here

    https://arxiv.org/pdf/1706.03762.pdf
"""

from __future__ import division, print_function

import keras
import numpy as np
from keras import backend as K
from keras import activations
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras.layers import Dense, Embedding, Input
from keras.models import Model

# TODO:
# - residual connections
# - share weight matrix in embedding layers
# - multiply weights in embedding layers by sqrt(d_model)
# - Mask decoder
# - hook up encoder/decoder correctly for training/inference


DEBUG = False
def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class MultiHeadAttention(Layer):
    def __init__(self, *, h, d_model, **kwargs):
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
    def __init__(self, d_model, d_k, d_v, activation, **kwargs):
        debug('init AttentionHead') 
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.scalar = np.sqrt(self.d_k)
        self.activation = activations.get(activation)
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


class LayerNorm(Layer):
    def __init__(self, x, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.supports_masking = True

    def call(self, inputs):
        return self.x + inputs

    def compute_output_shape(self, input_shape):
        return input_shape


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


def init_model(n_heads, encoder_layers, decoder_layers, d_model, vocab_size,
               sequence_len):

    # create input embedding
    input_input = Input(shape=(None,))
    input_embedding = Embedding(input_dim=vocab_size, output_dim=d_model,
                                input_length=sequence_len)(input_input)
    input_embedding = PositionalEncoding(d_model, sequence_len)(input_embedding)

    # make encoder
    encoder_layer_input = input_embedding
    for i in range(1, encoder_layers+1):
        names = iter([
            'encoder_subl%s_mha' % i,
            'encoder_subl%s_layernorm1' % i,
            'encoder_subl%s_ffn1' % i,
            'encoder_subl%s_ffn2' % i,
            'encoder_subl%s_layernorm2' % i,
        ])
        encoder = MultiHeadAttention(h=n_heads, d_model=d_model, name=next(names))(encoder_layer_input)
        encoder_sublayer1 = LayerNorm(encoder_layer_input, name=next(names))(encoder)
        encoder = Dense(d_model, activation='relu', name=next(names))(encoder_sublayer1)
        encoder = Dense(d_model, name=next(names))(encoder)
        encoder = LayerNorm(encoder_sublayer1, name=next(names))(encoder)
        encoder_layer_input = encoder
    # finally pull it all together in a model
    encoder_model = Model(inputs=input_input, outputs=encoder)

    # create output embedding
    output_input = Input(shape=(None,))
    output_embedding = Embedding(input_dim=vocab_size, output_dim=d_model,
                                 input_length=sequence_len)(output_input)
    output_embedding = PositionalEncoding(d_model, sequence_len)(output_embedding)

    # make decoder
    decoder_layer_input = output_embedding
    for i in range(1, decoder_layers+1):
        names = iter([
            'decoder_subl%s_mha1' % i,
            'decoder_subl%s_layernorm1' % i,
            'decoder_subl%s_mha2' % i,
            'decoder_subl%s_layernorm2' % i,
            'decoder_subl%s_ffn1' % i,
            'decoder_subl%s_ffn2' % i,
            'decoder_subl%s_layernorm3' % i,
        ])
        decoder = MultiHeadAttention(h=n_heads, d_model=d_model, name=next(names))(decoder_layer_input)
        decoder_sublayer1 = LayerNorm(decoder_layer_input, name=next(names))(decoder)
        decoder = MultiHeadAttention(h=n_heads, d_model=d_model, name=next(names))(encoder)
        decoder_sublayer2 = LayerNorm(decoder_sublayer1, name=next(names))(decoder)
        decoder = Dense(d_model, activation='relu', name=next(names))(decoder_sublayer2)
        decoder = Dense(d_model, name=next(names))(decoder)
        decoder = LayerNorm(decoder_sublayer2, name=next(names))(decoder)
        # output of layer becomes input of next layer
        decoder_layer_input = decoder
    # finally stack a linear transformation with softmax activation
    # to get next token probabilities
    decoder = Dense(d_model, activation='softmax')(decoder)
    decoder_model = Model(inputs=[input_input, output_input], outputs=decoder)

    return encoder_model, decoder_model


if __name__ == '__main__':
    n_heads = 8
    # # this is so cocnat(heads) has shape d_model
    # # (as each attention output has shape d_v)
    encoder_layers = decoder_layers = 6
    d_model = 64 * n_heads
    vocab_size = 30
    sequence_len = 30
    test_sequence_len = 100
    DEBUG = False

    encoder, decoder = init_model(
        n_heads=n_heads, encoder_layers=encoder_layers,
        decoder_layers=decoder_layers, d_model=d_model, vocab_size=vocab_size,
        sequence_len=sequence_len)

    # encoder.summary(line_length=100)
    decoder.summary(line_length=100)
    keras.utils.plot_model(encoder, 'encoder.dot')
    keras.utils.plot_model(decoder, 'decoder.dot')

    sequence = np.array(np.randint(a=0, b=vocab_size)
                        for _ in range(test_sequence_len))

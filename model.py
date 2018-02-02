"""
Implementation of Transformer model, as described here

    https://arxiv.org/pdf/1706.03762.pdf
"""

from __future__ import print_function, division

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
    encoder = input_embedding
    for _ in range(encoder_layers):
        encoder = MultiHeadAttention(h=n_heads, d_model=d_model)(encoder)
        encoder_sublayer1 = LayerNorm(input_embedding)(encoder)
        encoder = Dense(d_model, activation='relu')(encoder_sublayer1)
        encoder = Dense(d_model)(encoder)
        encoder = LayerNorm(encoder_sublayer1)(encoder)
        input_embedding = encoder

    # create output embedding
    output_input = Input(shape=(None,))
    output_embedding = Embedding(input_dim=vocab_size, output_dim=d_model,
                                 input_length=sequence_len)(output_input)
    output_embedding = PositionalEncoding(d_model, sequence_len)(output_embedding)

    # make decoder
    decoder = output_embedding
    for _ in range(decoder_layers):
        decoder = MultiHeadAttention(h=n_heads, d_model=d_model)(decoder)
        decoder_sublayer1 = LayerNorm(output_embedding)(decoder)
        decoder_sublayer2 = MultiHeadAttention(h=n_heads, d_model=d_model)(encoder)
        decoder = Dense(d_model, activation='relu')(decoder_sublayer2)
        decoder = Dense(d_model)(decoder)
        decoder = LayerNorm(decoder_sublayer2)(decoder)
        output_embedding = decoder  # correc?
    # finally stack a linear transformation with softmax activation
    # to get next token probabilities
    decoder = Dense(d_model, activation='softmax')(decoder)

    # finally pull it all together in a model
    encoder_model = Model(inputs=input_input, outputs=encoder)
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
    DEBUG = False

    encoder, decoder = init_model(
        n_heads=n_heads, encoder_layers=encoder_layers,
        decoder_layers=decoder_layers, d_model=d_model, vocab_size=vocab_size,
        sequence_len=sequence_len)

    encoder.summary()
    decoder.summary()

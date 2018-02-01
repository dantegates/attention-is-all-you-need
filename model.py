import numpy as np
from keras import backend as K
from keras import activations
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras.layers import Dense, Embedding, Input
from keras.models import Model

# https://arxiv.org/pdf/1706.03762.pdf
## Enccoder
# 1. Attention layer
# 2. multi-head attention
#    - project Q, K, V into d_k, d_k, d_v dimmensions respectively (heads)
#    - concat heads
#    - project concatenation
# 3. Add & Norm
# 4. Feed Forward
# 5. Add & Norm


DEBUG = True
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
        # _ = [head.build((self.d_model, self.d_k)) for head in self.heads]
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
        self.scalar = K.variable(np.sqrt(self.d_k))
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


if __name__ == '__main__':
    h = 8
    # this is so cocnat(heads) has shape d_model
    # (as each attention output has shape d_v)
    N = 6
    d_model = 64 * h
    vocab_size = 30
    sequence_len = 30
    DEBUG = False

    input_ = Input(shape=(None,))
    embedding = Embedding(input_dim=vocab_size, output_dim=d_model,
                          input_length=sequence_len)(input_)
    embedding = PositionalEncoding(d_model, sequence_len)(embedding)
    encoder = embedding
    for _ in range(N):
        encoder = MultiHeadAttention(h=h, d_model=d_model)(encoder)
        encoder = LayerNorm(embedding)(encoder)
        encoder = Dense(d_model, activation='relu')(encoder)
        encoder = Dense(d_model)(encoder)
        encoder = LayerNorm(embedding)(encoder)
    model = Model(inputs=input_, outputs=encoder)
    model.summary()

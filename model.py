import numpy as np
from keras import backend as K
from keras.engine.topology import Layer

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

class Attention(Layer):
    def __init__(self, d_k, d_v, **kwargs):
        self.d_k = d_k
        self.d_v = d_v
        self.attention_scalar = K.variable(np.sqrt(d_k))
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # perhaps these assertions are not very generic
        # leave in place while experimenting however
        assert input_shape[0] == self.d_k
        assert input_shape[0] == self.d_v
        self.K = self.add_weight(name='K',
                                 shape=(input_shape[0], self.d_k),
                                 initializer='uniform',
                                 trainable=True)
        self.V = self.add_weight(name='V',
                                 shape=(input_shape[0], self.d_v),
                                 initializer='uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)
    
    def call(self, Q):
        X = self.activation(K.dot(Q, K.transpose(self.K)))
        X /= self.d_k
        return K.dot(X, self.V)
    
    def compute_output_shape(self, input_shape):
        return (self.input_shape[0], self.d_v)
    

class MultiHeadAttention(Layer):
    def __init__(self, heads, **kwargs):
        self.heads = heads
        super(MultiHeadAttention, Layer).__init__(**kwargs)
        
    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
    
    def call(self, Q):
        return K.concatenate([head(Q) for head in self.heads])


if __name__ == '__main__':
    h = 8
    d_k = 64
    d_v = 64
    # this is so cocnat(heads) has shape d_model
    # (as each attention output has shape d_v)
    d_model = 64 * h

    attention_layers = [Attention(d_k=d_k, d_v=d_v, activation='softmax') for _ in range(h)]
    multi_head = MultiHeadAttention(attention_layers)

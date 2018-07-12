import unittest

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from model import AttentionHead, PositionalEncoding


def calc_positional_encoding(sequence_dim, d_model):
    rows, cols = np.indices((sequence_dim, d_model))
    numerator = np.where(cols % 2, np.cos(rows), np.sin(rows))
    denominator = (10_000**((2*cols)/d_model))
    return numerator / denominator


class TestPositionalEncoding(unittest.TestCase):
    def test(self):
        # batch dim / seq dim / embedding dim
        sequence_dim, d_model = 5, 32
        input_ = Input((None, d_model))
        positional_encoding = PositionalEncoding()(input_)
        model = Model(inputs=input_, outputs=positional_encoding)
        actual = model.predict(np.zeros((1, sequence_dim, d_model)))
        expected = calc_positional_encoding(sequence_dim, d_model)
        self.assertTrue(np.allclose(actual, expected))


class TestMasking(unittest.TestCase):
    def test(self):
        x = K.variable(np.array([[
            [1, 2, 3],
            [4, 5, 5],
            [7, 8, 9],
        ]]))
        actual = K.eval(AttentionHead.mask(x))
        expected = np.array([[
            [    1, -1e15, -1e15],
            [    4,     5, -1e15],
            [    7,     8,     9],
        ]])
        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest import mock

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from model import AttentionHead, PositionalEncoding, LayerNormalization


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


class TestLayerNormalization(unittest.TestCase):
    @mock.patch('model.LayerNormalization.build')
    def test_1(self, mock_build):
        X = K.variable(np.array([[
            [1, 2, 3],
            [0, 10, 7],
            [90, 100, 110],
        ]]))
        layer_norm = LayerNormalization()
        layer_norm.gain = 1
        layer_norm.bias = 0
        actual = K.eval(layer_norm(X))
        expected = np.array([[
            [-1.22474487,  0.        ,  1.22474487],
            [-1.35244738,  1.03422447,  0.31822291],
            [-1.22474487,  0.        ,  1.22474487]
        ]])
        self.assertTrue(np.allclose(actual, expected))

    @mock.patch('model.LayerNormalization.build')
    def test_2(self, mock_build):
        X = K.variable(np.array([[
            [1, 2, 3],
            [0, 10, 7],
            [90, 100, 110],
        ]]))
        layer_norm = LayerNormalization()
        layer_norm.gain = np.array([1, 0, 2])
        layer_norm.bias = np.array([0, 1, 2])
        actual = K.eval(layer_norm(X))
        expected = np.array([[
            [-1.22474487,  1.,   4.44948974],
            [-1.35244738,  1.,   2.63644582],
            [-1.22474487,  1.,   4.44948974]
        ]])
        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()

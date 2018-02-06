from functools import partial

import numpy as np


class TrainingData:
    BEGIN = '<start>'
    TERMINATE = '<end>'

    def __init__(self, text, max_len):
        self.text = text
        self.max_len = max_len
        self.tokens = [self.BEGIN]*max_len + list(text) + [self.TERMINATE]
        self.chars = sorted(set(self.tokens))
        self.vocab_size = len(self.chars)
        self.char_map, self.idx_map = self.init_maps(text)
        self.x, self.y = self.init_xy(text)

    def init_maps(self, text):
        char_map, idx_map = {}, {}
        for i, c in enumerate(self.chars):
            char_map[c] = i
            idx_map[i] = c
        return char_map, idx_map

    def init_xy(self, text):
        sentences = []
        next_sentences = []
        for i in range(0, len(self.tokens)-self.max_len):
            sentences.append(self.tokens[i:i+self.max_len])
            next_sentences.append(self.tokens[i+1:self.max_len+1])
        x = np.zeros((len(sentences), self.max_len), dtype=np.int64)
        y = np.zeros((len(sentences), self.max_len, self.vocab_size+1))
        for i, s in enumerate(sentences):
            for t, char in enumerate(s):
                x[i, t] = self.char_map[char]
                # y must be one hot encoded
                y[i, t, self.char_map[char]] = 1
        # offset trainint/test
        x = x[:-1]
        y = y[1:]
        return x, y


beatles_text = """Boy, you gotta carry that weight
Carry that weight a long time
Boy, you gonna carry that weight
Carry that weight a long time

I never give you my pillow
I only send you my invitation
And in the middle of the celebrations
I break down

Boy, you gotta carry that weight
Carry that weight a long time
Boy, you gotta carry that weight
You're gonna carry that weight a long time
"""


BEATLES = partial(TrainingData, beatles_text)

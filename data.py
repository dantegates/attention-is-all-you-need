from functools import partial
import glob
import os
import random
import string

import numpy as np
#import requests as rq


class TrainingData:
    START = '<start>'
    END = '<end>'
    UNKOWN = '<unk>'

    def __init__(self, directory, extension, sequence_len, batch_size, seed=None):
        self.sequence_len = sequence_len
        self.chars = sorted(set(string.printable))
        self.vocab_size = len(self.chars)
        self.char_map, self.idx_map = self.init_maps()
        self.directory = directory
        self.files = glob.glob(
            os.path.join(self.directory, '*%s' % extension))
        self.batch_size = batch_size
        self.seed = seed
        self.skipped = set()

    def __iter__(self):
        while True:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(self.files)
            for file in self.files:
                with open(file) as f:
                    content = f.read()
                if len(content) < self.batch_size:
                    if not file in self.skipped:
                        print('skipping', file)
                        self.skipped.add(file)
                    continue
                # pad content
                content = [self.START]*self.sequence_len \
                          + list(content) \
                          + [self.END]
                # kind of sloppy, just don't feel like writing batching
                # correctly at the moment
                slice_len = self.sequence_len + self.batch_size + 1
                i = random.randint(0, len(content) - slice_len)
                content = content[i:i+slice_len]
                x, y = self.init_xy(content)
                assert x.shape == (self.batch_size, self.sequence_len), 'unexpected x.shape %s' % (x.shape,)
                assert y.shape == (self.batch_size, self.sequence_len, self.vocab_size+1), 'unexpected y.shape %s' % (y.shape,)
                yield [x, x], y

    def init_maps(self):
        char_map, idx_map = {}, {}
        char_map[self.START] = 0
        idx_map[0] = self.START
        char_map[self.END] = 1
        idx_map[1] = self.END
        char_map[self.UNKOWN] = 2
        idx_map[2] = self.UNKOWN
        for i, c in enumerate(self.chars, start=3):
            char_map[c] = i
            idx_map[i] = c
        return char_map, idx_map

    def init_xy(self, text):
        sentences = []
        next_sentences = []
        for i in range(0, len(text)-self.sequence_len):
            sentences.append(text[i:i+self.sequence_len])
            next_sentences.append(text[i+1:self.sequence_len+1])
        x = np.zeros((len(sentences), self.sequence_len), dtype=np.int64)
        y = np.zeros((len(sentences), self.sequence_len, self.vocab_size+1))
        for i, s in enumerate(sentences):
            for t, char in enumerate(s):
                char = char if char in self.chars else self.UNKOWN
                x[i, t] = self.char_map[char]
                # y must be one hot encoded
                y[i, t, self.char_map[char]] = 1.0
        # offset train/test
        x = x[:-1] 
        y = y[1:]
        return x, y

BEATLES = partial(TrainingData, directory='beatles', extension='.txt')
CNN = partial(TrainingData, directory='cnn/**', extension='.story')

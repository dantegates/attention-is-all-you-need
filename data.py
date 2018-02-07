from functools import partial
import glob
import os
import random
import string

import numpy as np
import requests as rq


class TrainingData:
    BEGIN = '<start>'
    TERMINATE = '<end>'
    UNKOWN = '<unk>'

    def __init__(self, text, max_len):
        self.text = text
        self.max_len = max_len
        self.tokens = [self.BEGIN]*max_len + list(text) + [self.TERMINATE]
        self.chars = sorted(set(text)) + {self.BEGIN} + {self.TERMINATE} + {self.UNKOWN}
        self.vocab_size = len(self.chars)
        self.char_map, self.idx_map = self.init_maps()
        self.x, self.y = self.init_xy(text)

    def init_maps(self):
        char_map, idx_map = {}, {}
        char_map[self.BEGIN] = 0
        idx_map[0] = self.BEGIN
        char_map[self.TERMINATE] = 1
        idx_map[1] = self.TERMINATE
        char_map[self.UNKOWN] = 2
        idx_map[2] = self.UNKOWN
        for i, c in enumerate(self.chars, start=3):
            char_map[c] = i
            idx_map[i] = c
        return char_map, idx_map

    def init_xy(self, text):
        sentences = []
        next_sentences = []
        text = [self.BEGIN]*self.max_len + list(text) + [self.TERMINATE]
        for i in range(0, len(text)-self.max_len):
            sentences.append(text[i:i+self.max_len])
            next_sentences.append(text[i+1:self.max_len+1])
        x = np.zeros((len(sentences), self.max_len), dtype=np.int64)
        y = np.zeros((len(sentences), self.max_len, self.vocab_size+1))
        for i, s in enumerate(sentences):
            for t, char in enumerate(s):
                char = char if char in self.chars else self.UNKOWN
                x[i, t] = self.char_map[char]
                # y must be one hot encoded
                y[i, t, self.char_map[char]] = 1
        # offset trainint/test
        x = x[:-1] 
        y = y[1:]
        return x, y


class FileGeneratorTrainingData(TrainingData):
    def __init__(self, max_len, directory, extension, batch_size):
        self.max_len = max_len
        self.chars = sorted(set(string.printable))
        self.vocab_size = len(self.chars)
        self.char_map, self.idx_map = self.init_maps()
        self.directory = directory
        self.files = glob.glob(
            os.path.join(self.directory, '*%s' % extension))
        self.batch_size = batch_size
        self.skipped = set()

    def __iter__(self):
        while True:
            random.shuffle(self.files)
            for file in random.choices(self.files, k=100):
                with open(file) as f:
                    content = f.read()
                    if len(content) < self.max_len+self.batch_size:
                        if not file in self.skipped:
                            print('skipping file', file)
                            self.skipped.add(file)
                        continue 
                    i = random.randint(0, len(content) - self.max_len)
                    # sloppy hack as below
                    content = content[i:i+self.max_len]
                    x, y = self.init_xy(content)
                    # kind of sloppy, just don't feel like writing batching
                    # correctly at the moment
                    idx = np.random.randint(len(x), size=self.batch_size)
                    yield [x[idx], x[idx]], y[idx]


BEATLES = partial(FileGeneratorTrainingData, directory='beatles', extension='.txt')
CNN = partial(FileGeneratorTrainingData, directory='cnn/**', extension='.story')

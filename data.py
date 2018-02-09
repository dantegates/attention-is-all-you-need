import collections
from functools import partial
import glob
import os
import string

import numpy as np


class BatchGenerator:

    START = '<start>'
    END = '<end>'
    UNKOWN = '<unk>'
    CHARS = sorted(set(string.printable))

    CHAR_MAP = {
        START: 0,
        END: 1,
        UNKOWN: 2,
    }

    IDX_MAP = {i: c for c, i in CHAR_MAP.items()}


    CHAR_MAP.update((c, i) for i, c in enumerate(CHARS, start=max(IDX_MAP+1))
    IDX_MAP.update((i, c) for i, c in enumerate(CHARS, start=max(IDX_MAP+1)))

    VOCAB_SIZE = len(CHARS)

    def __init__(self, encoder_len, decoder_len, directory, extension, batch_size,
                 step_size, tokenizer='chars'):
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.directory = directory
        self.files = glob.glob(os.path.join(self.directory, '*%s' % extension))
        self.batch_size = batch_size
        self.step_size = step_size
        self.n_batches = sum(1 for _ in self.fetch_examples()) // self.batch_size
        self.tokenizer = tokenizer
        self.skipped = set()

    def __iter__(self):
        while True:
            x1 = np.zeros((self.batch_size+1, self.encoder_len))
            x2 = np.zeros((self.batch_size+1, self.decoder_len))
            y = np.zeros((self.batch_size+1, self.decoder_len, self.VOCAB_SIZE+1))
            for i, (x1_content, x2_content) in enumerate(self.fetch_examples()):
                i = i % self.batch_size
                x1[i, :] = self.make_x(x1_content[-self.encoder_len:], encoder=True)
                x2[i, :] = self.make_x(x2_content, encoder=False)
                y[i+1,:,:] = self.make_y(x2_content)
                if i == self.batch_size-1:
                    # offset x/y
                    yield [x1[:-1], x2[:-1]], y[1:]
                    x1 = np.zeros((self.batch_size+1, self.encoder_len))
                    x2 = np.zeros((self.batch_size+1, self.decoder_len))
                    y = np.zeros((self.batch_size+1, self.decoder_len, self.VOCAB_SIZE+1))

    def fetch_content(self):
        for file in self.files:
            with open(file) as f:
                content = f.read()
                if len(content) < self.batch_size:
                    if not file in self.skipped:
                        print('skipping', file)
                        self.skipped.add(file)
                yield content

    def fetch_examples(self):
        for content in self.fetch_content():
            lines = content.split('\n')
            for i in range(len(lines)):
                l1, l2 = lines[:i], lines[i]
                for j in range(0, len(l2)+1, self.step_size):
                    yield '\n'.join(l1), l2[:j] + '\n'

    def tokenize(self, text):
        tokens = []
        if self.tokenizer == 'chars':
            tokens = list(text)
        # if self.tokenizer == 'words':
        #     for p in string.punctuation:
        #         text = text.replace(p, ' %s ' % p)
        #     tokens = [s.strip() for s in text.split()]
        tokens = [c if c in self.CHARS else self.UNKOWN
                  for c in tokens]
        return collections.deque(tokens)

    def make_x(self, text, encoder):
        sequence_len = self.encoder_len if encoder else self.decoder_len
        tokens = self.tokenize(text)
        if len(tokens) < sequence_len:
            tokens.append(self.END)
        while len(tokens) < sequence_len:
            tokens.appendleft(self.START)
        x = np.array([self.CHAR_MAP[c] for c in tokens])
        return x

    def make_y(self, text):
        sequence_len = self.decoder_len
        tokens = self.tokenize(text)
        if len(tokens) < sequence_len:
            tokens.append(self.END)
        while len(tokens) < sequence_len:
            tokens.appendleft(self.START)
        y = np.zeros((sequence_len, self.VOCAB_SIZE+1))
        for i, c in enumerate(tokens):
            y[i][self.CHAR_MAP[c]] = 1
        return y


BEATLES = partial(BatchGenerator, directory='beatles', extension='.txt')
CNN = partial(BatchGenerator, directory='cnn/**', extension='.story')

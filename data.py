import bisect
import collections
import glob
import logging
import os
import random
import string
import itertools
from functools import partial

import numpy as np


logger = logging.getLogger(__name__)


class BatchGenerator:
    PAD = '<pad>'
    END = '\a'  # this needs to be a single ascii character to make
                # tokenizing simple
    UNKOWN = '<unk>'

    def __init__(self, sequence_len, directory, extension, batch_size,
                 step_size, vocab_size=None, tokenizer='chars'):
        self.sequence_len = sequence_len
        self.directory = directory
        self.files = glob.glob(os.path.join(self.directory, '*%s' % extension))
        self.batch_size = batch_size
        self.step_size = step_size
        self.tokenizer = tokenizer

        self.corpi = self.tokenize_corpi()
        self.example_map = list(itertools.accumulate(len(tokens) for tokens in self.corpi))
        self.n_examples = self.example_map[-1]
        self.n_batches = self.n_examples // self.batch_size
        
        tokens = self.init_tokens(vocab_size)
        self.char_map = {
            self.PAD: 0,
            self.END: 1,
            self.UNKOWN: 2,
        }
        self.idx_map = {i: c for c, i in self.char_map.items()}
        self.char_map.update((c, i) for i, c in enumerate(tokens, start=max(self.idx_map)+1))
        self.idx_map.update((i, c) for c, i in self.char_map.items())
        self.tokens = sorted(self.char_map)
        self.vocab_size = len(self.tokens)

        self.skipped = set()

    def __iter__(self):
        while True:
            x1 = np.zeros((self.batch_size, self.sequence_len))
            x2 = np.zeros((self.batch_size, self.sequence_len))
            y = np.zeros((self.batch_size, self.sequence_len, self.vocab_size+1))
            for i, (ex1, ex2, target) in enumerate(self.fetch_examples()):
                i = i % self.batch_size
                x1[i, :] = self.tokens_to_x(ex1)
                x2[i, :] = self.tokens_to_x(ex2)
                y[i,:,:] = self.tokens_to_y(target)
                if i == self.batch_size-1:
                    yield [x1, x2], y
                    x1 = np.zeros((self.batch_size, self.sequence_len))
                    x2 = np.zeros((self.batch_size, self.sequence_len))
                    y = np.zeros((self.batch_size, self.sequence_len, self.vocab_size+1))
                    
    def fetch_file_content(self):
        for file in self.files:
            with open(file) as f:
                content = f.read() + self.END
                if len(content) < self.batch_size:
                    if not file in self.skipped:
                        print('skipping', file)
                        self.skipped.add(file)
                yield content

    def fetch_examples(self):
        positions = list(range(0, self.n_examples, self.step_size))
        np.random.shuffle(positions)
        for p in positions:
            corpus_pos = bisect.bisect_right(self.example_map, p)
            corpus = self.corpi[corpus_pos]
            i = p - self.example_map[corpus_pos]
            start = i - self.sequence_len
            x1 = corpus[start:i]
            x2 = corpus[start:i]
            y = corpus[start+1:i+1]
            yield x1, x2, y

    def tokenize_corpi(self):
        tokens = []
        for file_content in self.fetch_file_content():
            tokens.append(self.tokenize(file_content))
        return tokens

    def tokenize(self, text):
        tokens = []
        if self.tokenizer == 'chars':
            tokens = list(text)
        elif self.tokenizer == 'words':
            for p in string.punctuation.replace("'", ''):
                text = text.replace(p, ' %s ' % p)
            text = text.replace('\n', ' \n ')
            tokens = [s.strip(' ') for s in text.split(' ') if s.strip(' ')]
            tokens = list(itertools.chain(tokens))
        else:
            raise ValueError('unrecognized tokenizer')
        return tokens

    def tokens_to_x(self, tokens):
        while len(tokens) < self.sequence_len:
            tokens = [self.PAD] + tokens
        logger.debug('x tokens: %r', tokens)
        x = np.array([self.char_map[c] if c in self.char_map else self.char_map[self.UNKOWN]
                      for c in tokens])
        return x

    def tokens_to_y(self, tokens):
        while len(tokens) < self.sequence_len:
            tokens = [self.PAD] + tokens
        y = np.zeros((self.sequence_len, self.vocab_size+1))
        logger.debug('y tokens: %r', tokens)
        for i, c in enumerate(tokens):
            idx = self.char_map[c] if c in self.char_map else self.char_map[self.UNKOWN]
            y[i][idx] = 1
        return y

    def idx_to_char(self, idx):
        return self.idx_map[idx] if idx in self.idx_map else self.UNKOWN

    def init_tokens(self, maxsize):
        all_tokens = collections.Counter()
        for corpus in self.corpi:
            all_tokens.update(corpus)
        return sorted([item for item, count in all_tokens.most_common(maxsize)])


LYRICS = partial(BatchGenerator, directory='lyrics', extension='.txt')
BEATLES = partial(BatchGenerator, directory='beatles', extension='.txt')
CNN = partial(BatchGenerator, directory='cnn/**', extension='.story')

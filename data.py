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
                 step_size, tokenizer='chars'):
        self.sequence_len = sequence_len
        self.directory = directory
        self.files = glob.glob(os.path.join(self.directory, '*%s' % extension))
        self.batch_size = batch_size
        self.step_size = step_size
        self.tokenizer = tokenizer

        self.examples = self.fetch_examples()
        self.test_example = self.examples[10]  # before shuffle
        self.n_batches = (len(self.examples)-1) // self.batch_size
        
        tokens = self.init_tokens()
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
            random.shuffle(self.examples)
            x1 = np.zeros((self.batch_size, self.sequence_len))
            x2 = np.zeros((self.batch_size, self.sequence_len))
            y = np.zeros((self.batch_size, self.sequence_len, self.vocab_size+1))
            for i, (ex1, ex2, target) in enumerate(self.examples):
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
        examples = []
        for file_content in self.fetch_file_content():
            lines = file_content.split('\n')
            for i in range(len(lines)):
                context_text, target_text = '\n'.join(lines[:i]), lines[i] + '\n'
                if context_text:
                    context_text += '\n'
                x1 = self.tokenize(context_text)[-self.sequence_len:]
                target_tokens = self.tokenize(target_text)
                for j in range(0, len(target_tokens), self.step_size):
                    x2 = target_tokens[:j]
                    y = target_tokens[:j+1]
                    examples.append((x1, x2, y))
        return examples

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
        x = np.array([self.char_map[c] for c in tokens])
        return x

    def tokens_to_y(self, tokens):
        while len(tokens) < self.sequence_len:
            tokens = [self.PAD] + tokens
        y = np.zeros((self.sequence_len, self.vocab_size+1))
        logger.debug('y tokens: %r', tokens)
        for i, c in enumerate(tokens):
            idx = self.char_map[c]
            y[i][idx] = 1
        return y

    def idx_to_char(self, idx):
        return self.idx_map[idx] if idx in self.idx_map else self.UNKOWN

    def init_tokens(self):
        tokens = set()
        for x1, x2, y in self.examples:
            tokens.update(x1)
            tokens.update(x2)
            tokens.update(y)
        return sorted(tokens)


BEATLES = partial(BatchGenerator, directory='beatles', extension='.txt')
CNN = partial(BatchGenerator, directory='cnn/**', extension='.story')

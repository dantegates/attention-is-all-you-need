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

    def __init__(self, encoder_len, decoder_len, directory, extension, batch_size,
                 step_size, tokenizer='chars'):
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.directory = directory
        self.files = glob.glob(os.path.join(self.directory, '*%s' % extension))
        self.batch_size = batch_size
        self.step_size = step_size
        self.tokenizer = tokenizer

        self.examples = self.fetch_examples()
        self.n_batches = (len(self.examples)-1) // self.batch_size
        self.tokens = self.init_tokens()

        self.char_map = {
            self.PAD: 0,
            self.END: 1,
            self.UNKOWN: 2,
        }
        self.idx_map = {i: c for c, i in self.char_map.items()}
        self.char_map.update((c, i) for i, c in enumerate(self.tokens, start=max(self.idx_map)+1))
        self.idx_map.update((i, c) for i, c in enumerate(self.tokens, start=max(self.idx_map)+1))
        self.vocab_size = len(self.char_map)

        self.skipped = set()

    def __iter__(self):
        while True:
            random.shuffle(self.examples)
            x1 = np.zeros((self.batch_size+1, self.encoder_len))
            x2 = np.zeros((self.batch_size+1, self.decoder_len))
            y = np.zeros((self.batch_size+1, self.decoder_len, self.vocab_size+1))
            for i, (context, target) in enumerate(self.examples):
                i = i % self.batch_size
                x1[i, :] = self.tokens_to_x(context)
                x2[i, :] = self.tokens_to_x(target)
                y[i+1,:,:] = self.tokens_to_y(target)
                if i == self.batch_size-1:
                    # offset x/y
                    yield [x1[:-1], x2[:-1]], y[1:]
                    x1 = np.zeros((self.batch_size+1, self.encoder_len))
                    x2 = np.zeros((self.batch_size+1, self.decoder_len))
                    y = np.zeros((self.batch_size+1, self.decoder_len, self.vocab_size+1))

    def fetch_content(self):
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
        for content in self.fetch_content():
            lines = content.split('\n')
            for i in range(len(lines)):
                context_text, target_text = '\n'.join(lines[:i]), lines[i] + '\n'
                context_tokens = self.tokenize(context_text)
                context = context_tokens[-self.encoder_len:]
                target_tokens = self.tokenize(target_text)
                for j in range(1, len(target_tokens)+1, self.step_size):
                    target = target_tokens[:j]
                    logger.debug('fetched context: %r', context)
                    logger.debug('feteched target: %r', target)
                    examples.append((context, target))
        return examples

    def tokenize(self, text):
        tokens = []
        if self.tokenizer == 'chars':
            tokens = list(text)
        elif self.tokenizer == 'words':
            for p in string.punctuation.replace("'", ''):
                text = text.replace(p, ' %s ' % p)
            text = text.replace('\n', ' \n ')
            tokens = [s.strip(' ') for s in text.split(' ')]
            tokens = list(itertools.chain(tokens))
        else:
            raise ValueError('unrecognized tokenizer')
        return tokens

    def tokens_to_x(self, tokens):
        tokens = collections.deque(tokens)
        sequence_len = self.encoder_len
        while len(tokens) < sequence_len:
            tokens.appendleft(self.PAD)
        logger.debug('x tokens: %r', tokens)
        x = np.array([self.char_map[c] for c in tokens])
        return x

    def tokens_to_y(self, tokens):
        tokens = collections.deque(tokens)
        sequence_len = self.decoder_len
        while len(tokens) < sequence_len:
            tokens.appendleft(self.PAD)
        y = np.zeros((sequence_len, self.vocab_size+1))
        logger.debug('y tokens: %r', tokens)
        for i, c in enumerate(tokens):
            idx = self.char_map[c]
            y[i][idx] = 1
        return y

    def idx_to_char(self, idx):
        return self.idx_map[idx] if idx in self.idx_map else self.UNKOWN

    def init_tokens(self):
        tokens = set()
        for context, target in self.examples:
            tokens.update(context)
            tokens.update(target)
        return sorted(tokens)


BEATLES = partial(BatchGenerator, directory='beatles', extension='.txt')
CNN = partial(BatchGenerator, directory='cnn/**', extension='.story')

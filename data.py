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
                 step_size=1, delimiter=None, max_vocab_size=None, tokenizer='chars'):
        self.sequence_len = sequence_len
        self.directory = directory
        self.files = glob.glob(os.path.join(self.directory, '*%s' % extension))
        self.batch_size = batch_size
        self.step_size = step_size
        self.delimiter = delimiter
        self.tokenizer = tokenizer

        self.encoder_corpi, self.decoder_corpi = self.tokenize_corpi()
        self.example_map = list(itertools.accumulate(len(tokens)
                                for tokens in self.decoder_corpi))
        self.n_examples = self.example_map[-1]
        self.n_batches = self.n_examples // self.batch_size
        
        tokens = self.init_tokens(max_vocab_size)
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
                        continue
                if self.delimiter is not None:
                    encoder_input, decoder_input = content.split(self.delimiter)
                else:
                    encoder_input = decoder_input = content
                yield encoder_input, decoder_input

    def fetch_examples(self):
        example_positions = list(range(0, self.n_examples, self.step_size))
        np.random.shuffle(example_positions)
        for example_position in example_positions:
            # This gives us the index for the "bucket" containing the
            # target example. Note the encoder and decoder index is the
            # same.
            idx_bucket = bisect.bisect_right(self.example_map, example_position)
            encoder_tokens = self.encoder_corpi[idx_bucket]
            decoder_tokens = self.decoder_corpi[idx_bucket]

            # this is the index in the decoder tokens pointing to the
            # example determined by p
            p = 0 if idx_bucket == 0 else self.example_map[idx_bucket - 1]
            idx_target_token = example_position - p
            x2_start = max(0, idx_target_token - self.sequence_len)
            x2_stop = idx_target_token
            if encoder_tokens == decoder_tokens:
                x1_start, x1_stop = x2_start, x2_stop
            else:
                x1_start, x1_stop = 0, self.sequence_len
            x1 = encoder_tokens[x1_start:x1_stop]
            x2 = decoder_tokens[x2_start:x2_stop]
            y = decoder_tokens[x2_start+1:x2_stop+1]
            yield x1, x2, y

    def tokenize_corpi(self):
        encoder_tokens = []
        decoder_tokens = []
        for file_content in self.fetch_file_content():
            encoder_input, decoder_input = file_content
            encoder_tokens.append(self.tokenize(encoder_input))
            decoder_tokens.append(self.tokenize(decoder_input))
        return encoder_tokens, decoder_tokens

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
        for corpus in itertools.chain(self.encoder_corpi, self.decoder_corpi):
            all_tokens.update(corpus)
        return sorted([item for item, count in all_tokens.most_common(maxsize)])


LYRICS_TRAIN = partial(BatchGenerator, directory='lyrics-train', extension='.txt')
LYRICS_TEST = partial(BatchGenerator, directory='lyrics-test', extension='.txt')
BEATLES = partial(BatchGenerator, directory='beatles', extension='.txt')
CNN = partial(BatchGenerator, directory='summaries', extension='.story', delimiter='\t')
SONGNAMES_TRAIN = partial(BatchGenerator, directory='songnames-train', extension='.txt', delimiter='\t')
SONGNAMES_TEST = partial(BatchGenerator, directory='songnames-test', extension='.txt', delimiter='\t')

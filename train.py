import fnmatch
import os
import sys

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from model import init_model


CONTEXT_LENGTH = 200
BATCH_SIZE = 100
EPOCHS = 1
DEBUG = False


IGNORE_PATHS = ['site-packages', os.getcwd(), 'mac', 'bin']
PY_FILES = []
SEARCH_PATHS = filter(lambda p: not any(ignore in p for ignore in IGNORE_PATHS), sys.path)
for path in SEARCH_PATHS:
    if DEBUG:
        print('searching in', path)
    for root, dirnames, filenames in os.walk(path):
        if not 'site-package' in root:
            for filename in fnmatch.filter(filenames, '*.py'):
                dest = os.path.join(root, filename)
                PY_FILES.append(dest)


class Batcher:
    def __init__(self, context_size, step_size, files):
        self.context_size = context_size
        self.files = files
        self.step_size = step_size
        charset = set(self.gen_corpus(files))
        self.charset_size = len(charset)
        self._dv = DictVectorizer(sparse=False)
        self._dv.fit([{'c': c} for c in charset])

    def __iter__(self):
        for file in self.files:
            content = self.gen_text(file)
            if len(content) < self.context_size:
                continue
            V = self._dv.transform([{'c': c} for c in content])
            n_samples = len(content) - self.context_size
            X = np.zeros((n_samples, self.context_size, self.charset_size))
            y = np.zeros((n_samples, self.charset_size))
            for i in range(0, n_samples, self.step_size):
                X[i] = V[i:i+self.context_size]
                y[i] = V[i+self.context_size]
            yield X, y

    def gen_text(self, file):
        with open(file) as f:
            out = []
            try:
                contents = f.read()
            except UnicodeDecodeError:
                if DEBUG:
                    print('ignoring', file, 'unicode error')
            else:
                out = ['<start>' for _ in range(CONTEXT_LENGTH)]
                out += list(contents)
                out.append('<end>')
            return out

    def gen_corpus(self, files):
        return (c for f in files
                  for text in self.gen_text(f)
                  for c in text)

    @property
    def char_map(self):
        return {v: k.partition('c=')[-1] for k, v in self._dv.vocabulary_.items()}


if __name__ == '__main__':
    batcher = Batcher(CONTEXT_LENGTH, 1, PY_FILES[:100])
    train = list(batcher)
    Xs, ys = zip(*train)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    encoder, decoder = init_model(h=8, encoder_layers=6, decoder_layers=6)
    decoder.compile(loss='categorical_crossentropy', optimizer='adam')
    decoder.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

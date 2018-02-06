import numpy as np


text = """Boy, you gotta carry that weight
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
You're gonna carry that weight along time
"""


def training_data(max_len):
    tokens = ['<start>']*max_len + list(text) + ['<end>']
    chars = sorted(set(tokens))
    vocab_size = len(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    sentences = []
    next_chars = []
    for i in range(0, len(tokens) - max_len):
        sentences.append(tokens[i:i+max_len])
        next_chars.append(tokens[i+1:max_len+1])
    x = np.zeros((len(sentences), max_len), dtype=np.int64)
    y = np.zeros((len(sentences), max_len, vocab_size+1))
    for i, s in enumerate(sentences):
        for t, char in enumerate(s):
            x[i, t] = char_indices[char]
            y[i, t, char_indices[char]] = 1
    # offset training/test
    x = x[:-1]
    y = y[1:]
    return x, y, vocab_size

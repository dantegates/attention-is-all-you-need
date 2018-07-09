import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from data import BaseBatchGenerator


class TrainingExample:
    """Simple container to keep track of training data. Useful for debugging."""
    def __init__(self, item, context_text, target_text, context_tokens,
                 target_tokens):
        self.item = item
        self.context_text = context_text
        self.target_text = target_text
        self.context_tokens = context_tokens
        self.target_tokens = target_tokens


def load_files(files, tokenizer):
    """Load and tokenize files."""
    training_examples = []
    for file in files:
        context_text, target_text = self.load_file_contents(file)
        context_tokens = self.tokenizer(context_text)
        target_tokens = self.tokenizer(target_text)
        example = TrainingExample(file, context_text, target_text,
                                  context_tokens, target_tokens)
        training_examples.append(example)
    return training_examples


class SummaryBatchGenerator(BaseBatchGenerator):
    def __init__(self, max_context_len, max_target_len, eos_token):
        self.max_context_len = max_context_len
        self.max_target_len = max_target_len
        self.eos_token = eos_token

    def generate_steps(self, item):
        example = item  # simple alias
        if len(example.target_tokens) > self.max_target_len:
            return []
        encoder_tokens = example.context_tokens[:self.max_context_len]
        # see
        # https://github.com/tensorflow/tensor2tensor/blob/ea576658c608d8b805bbe64c1c85814a96b879b9/tensor2tensor/layers/common_hparams.py#L199
        decoder_tokens = example.target_tokens \
                       + [self.eos_token] \
                       + training_example.target_tokens \
                       + [self.eos_token]
        training_step = self.pad(encoder_tokens), self.pad(decoder_tokens)
        return [training_step]

    def generate_batches(self, steps, batch_size):
        batches = []
        current_batch = []
        current_batch_size = 0
        for i, training_step in enumerate(steps):
            encoder_tokens, decoder_tokens = training_step
            training_step_size = len(encoder_tokens) + len(decoder_tokens)
            if current_batch_size + training_step_size <= batch_size:
                current_batch_size += training_step_size
        return batches, steps[i:]

    def pad(self, tokens, prepend=False):
        # add 1 to sentence_len since we shift output one step forward to prevent
        # model from attending to future time steps
        tokens = pad_sequences(
            [tokens], maxlen=self.sentence_len, padding='post',
            truncating='post', value=self.pad_token)
        if prepend:
            tokens = pad_sequences(
                [tokens[0]], maxlen=self.sentence_len+1, padding='pre',
                value=self.pad_token)
        return tokens[0]







    def generate_batches(self, steps, batch_size, n_batches):
        for i in range(n_batches):
            start, stop = i*batch_size, (i+1)*batch_size
            batch_steps = steps[start:stop]
            encoder_steps, decoder_steps = zip(*batch_steps)
            x1 = np.array(encoder_steps)
            x2 = np.array(decoder_steps)
            # offset target from decoder input
            X = [x1, x2[:,:-1]]
            y = x2[:,1:]
            yield X, y



    
    def to_arrays(self, items, batch_size):
        x1s, x2s, ys = [], [], []
        for X, y in self.generate_epoch(items, batch_size):
            x1, x2 = X
            x1s.append(x1)
            x2s.append(x2)
            ys.append(y)
        X = [np.concatenate(x1s), np.concatenate(x2s)]
        y = np.concatenate(ys)
        return X, y

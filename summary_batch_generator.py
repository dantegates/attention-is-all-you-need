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
        with open(file) as f:
            context_text, target_text = f.read().split('\t')
        context_tokens = tokenizer(context_text)
        target_tokens = tokenizer(target_text)
        example = TrainingExample(file, context_text, target_text,
                                  context_tokens, target_tokens)
        training_examples.append(example)
    return training_examples


class SummaryBatchGenerator(BaseBatchGenerator):
    def __init__(self, max_context_len=None, max_target_len=None, eos_token=-1,
                 pad_token=0, prepend=True):
        self.max_context_len = max_context_len
        self.max_target_len = max_target_len
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.prepend = prepend

    def generate_steps(self, item):
        example = item  # alias
        if self.max_target_len is not None \
                and len(example.target_tokens) > self.max_target_len:
            return []
        if self.max_context_len is not None:
            encoder_tokens = example.context_tokens[:self.max_context_len]
        else:
            encoder_tokens = example.context_tokens
        decoder_tokens = example.target_tokens + [self.eos_token] if self.prepend else []
        decoder_tokens += example.target_tokens + [self.eos_token]
        training_step = encoder_tokens, decoder_tokens
        return [training_step]

    def generate_batches(self, steps, batch_size):
        batches = []
        min_batch_size = 0.95 * batch_size
        max_batch_size = 1.05 * batch_size
        step_sizes = [len(e_toks) + len(d_toks) for e_toks, d_toks in steps]
        current_batch_x1s = []
        current_batch_x2s = []
        current_batch_size = 0
        i = None
        items = enumerate(zip(steps, step_sizes, step_sizes[1:]))
        for i, (step, step_size, next_step_size) in items:
            if step_size > max_batch_size:
                print(f'skipping step with size {step_size}')
                continue
            encoder_tokens, decoder_tokens = step
            current_batch_x1s.append(encoder_tokens)
            current_batch_x2s.append(decoder_tokens)
            current_batch_size += step_size
            if min_batch_size <= current_batch_size + next_step_size <= max_batch_size:
                x1 = pad_sequences(current_batch_x1s, value=self.pad_token)
                x2 = pad_sequences(current_batch_x2s, value=self.pad_token)
                X = [x1, x2[:,:-1]]
                y = x2[:,1:]
                batches.append((X, y))
                current_batch_size = 0
                current_batch_x1s, current_batch_x2s = [], []
            # if there aren't enough steps left to create a full sized batch
            # then break
            if sum(step_sizes[i+1:]) < batch_size:
                break
        return (batches, steps[i+1:]) if i is not None else (batches, steps)

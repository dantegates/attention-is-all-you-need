from collections import deque
import os

import keras
from keras.callbacks import LambdaCallback, LearningRateScheduler, TerminateOnNaN
import numpy as np

from data import BEATLES, CNN
from model import Transformer



# model params
n_heads = 8
encoder_layers = decoder_layers = 2
d_model = 64 * n_heads
sequence_len = 100
layer_normalization = True
dropout = True
residual_connections = True

# training params
epochs = 100
batch_size = 30
warmup_steps = 1000
optimizer = keras.optimizers.adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
logfile = 'train.log'

training_data = BEATLES(sequence_len=sequence_len, batch_size=batch_size, seed=0)
vocab_size = training_data.vocab_size + 1

model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len,
        layer_normalization=layer_normalization, dropout=dropout,
        residual_connections=residual_connections)


def generate_text(epoch, logs, mode='random'):
    i = np.random.randint(vocab_size)
    x = np.zeros((1, sequence_len))
    terminate = training_data.char_map[training_data.END]
    next_idx = -1
    text = ''.join(training_data.idx_map[i] for i in x[0])
    while next_idx != terminate and len(text) < 2000:
        pred = model.predict([x, x])
        probs = pred[0][-1]
        next_idx = np.random.choice(range(len(probs)), p=probs)
        text += training_data.idx_map[next_idx]
        # shift elements backward
        x = np.roll(x, -1)
        x[0, -1] = next_idx
    remove = [training_data.START, training_data.END, training_data.UNKOWN]
    for rm in remove:
        text = text.replace(rm, '')
    with open(logfile, 'a') as f:
        f.write('epoch: %d, loss=%s\n' % (epoch, logs['loss']))
        f.write(text)
        f.write('\n\n')
    print(repr(text))

def lr_schedule(epoch):
    epoch += 1
    lr = d_model**-.5 * min(epoch**-.5, epoch*warmup_steps**-1.5)
    return lr

callbacks = []
callbacks.append(LambdaCallback(on_epoch_end=generate_text))
callbacks.append(LearningRateScheduler(lr_schedule))
callbacks.append(TerminateOnNaN())

# for debugging. e.g. if loss turns to NaN, batches[0] will contain batch
# that caused the NaN
batches = deque(maxlen=2)
def gen():
    for i in training_data:
        (x, x), y = i
        if np.isnan(x).any():
            raise Exception('NaNs in input')
        if np.isnan(y).any():
            raise Exception('NaNs in output')
        batches.append(i)
        yield i
gen = gen()


if __name__ == '__main__':
    if os.path.exists(logfile):
        # clear log file
        with open(logfile, 'w'):
            pass
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit_generator(gen, steps_per_epoch=len(training_data.files)*10,
                        epochs=epochs, callbacks=callbacks)
    model.save('model.h5')

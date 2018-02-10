import os
from collections import deque

import keras
import numpy as np
from data import BEATLES, CNN
from keras.callbacks import (LambdaCallback, LearningRateScheduler,
                             TerminateOnNaN)
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
step_size = 1
tokenizer = 'words'
training_data = BEATLES(encoder_len=sequence_len, decoder_len=sequence_len,
                        batch_size=batch_size, step_size=step_size,
                        tokenizer=tokenizer)
vocab_size = training_data.vocab_size + 1

model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len,
        layer_normalization=layer_normalization, dropout=dropout,
        residual_connections=residual_connections)


def generate_text(epoch, logs):
    x1 = training_data.text_to_x('', encoder=True).reshape((1, -1))
    x2 = training_data.text_to_x('', encoder=False).reshape((1, -1))
    x = [x1, x2]
    char = -1
    text = ''
    while char != training_data.END and len(text) < 2000:
        pred = model.predict(x)
        probs = pred[0][-1]
        idx = np.random.choice(range(len(probs)), p=probs)
        char = training_data.idx_to_char(idx)
        text += char
        # shift elements backward
        x1, x2 = x
        x2 = np.roll(x2, -1)
        x2[0, -1] = idx
        x = [x1, x2]
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
gen = (i for i in training_data)


if __name__ == '__main__':
    if os.path.exists(logfile):
        # clear log file
        with open(logfile, 'w'):
            pass
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    try:
        model.fit_generator(gen, steps_per_epoch=len(training_data.files)*10,
                            epochs=epochs, callbacks=callbacks)
    except KeyboardInterrupt:
        pass
    model.save('model.h5')

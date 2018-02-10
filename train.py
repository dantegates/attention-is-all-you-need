import os
from collections import deque

import keras
import numpy as np
from data import BEATLES, CNN
from keras.callbacks import (LambdaCallback, LearningRateScheduler,
                             TerminateOnNaN, ModelCheckpoint)
from model import Transformer

# model params
n_heads = 8
encoder_layers = decoder_layers = 4
d_model = 64 * n_heads
sequence_len = 150
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
batch_generator = BEATLES(encoder_len=sequence_len, decoder_len=sequence_len,
                        batch_size=batch_size, step_size=step_size,
                        tokenizer=tokenizer)
vocab_size = batch_generator.vocab_size + 1

model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len,
        layer_normalization=layer_normalization, dropout=dropout,
        residual_connections=residual_connections)


def generate_text(epoch, logs):
    x1 = batch_generator.tokens_to_x([]).reshape((1, -1))
    x2 = batch_generator.tokens_to_x([]).reshape((1, -1))
    x = [x1, x2]
    token = object()
    tokens = []
    while token != batch_generator.END and len(tokens) < sequence_len:
        # predict and sample an index according to probability dist.
        pred = model.predict(x)
        probs = pred[0][-1]
        idx = np.random.choice(range(len(probs)), p=probs)

        # convert the index to token
        # The model is trained on context of all previous lines.
        # Therefore if token is a newline, reinitialize the context (x1)
        # and decoder input (x2).
        #
        # Otherwise, add idx to the decoder input and leave the encoder
        # context as is
        token = batch_generator.idx_to_char(idx)
        tokens.append(token)
        if token == '\n':
            x1 = batch_generator.tokens_to_x(tokens)
            x2 = batch_generator.tokens_to_x([])
        else:
            # shift elements backward
            x1, x2 = x
            x2 = np.roll(x2, -1)
            x2[0, -1] = idx
        x = [x1, x2]

    # remove special tokens
    remove = [batch_generator.START, batch_generator.END, batch_generator.UNKOWN]
    text = ' '.join(t for t in tokens if not t in remove)
    with open(logfile, 'a') as f:
        f.write('epoch: %d, loss=%s\n' % (epoch, logs['loss']))
        f.write(text)
        f.write('\n\n')

def lr_schedule(epoch):
    epoch += 1
    lr = d_model**-.5 * min(epoch**-.5, epoch*warmup_steps**-1.5)
    return lr

callbacks = []
#callbacks.append(LambdaCallback(on_epoch_end=generate_text))
callbacks.append(LearningRateScheduler(lr_schedule))
callbacks.append(TerminateOnNaN())
callbacks.append(ModelCheckpoint(filepath='model.h5'))

# for debugging. e.g. if loss turns to NaN, batches[0] will contain batch
# that caused the NaN
batches = deque(maxlen=2)
gen = (i for i in batch_generator)


if __name__ == '__main__':
    if os.path.exists(logfile):
        # clear log file
        with open(logfile, 'w'):
            pass
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    try:
        model.fit_generator(gen, steps_per_epoch=len(batch_generator.files)*10,
                            epochs=epochs, callbacks=callbacks)
    except KeyboardInterrupt:
        pass
model.save('model.h5')

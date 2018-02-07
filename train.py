import numpy as np
from data import BEATLES, CNN
import keras
from keras.callbacks import LambdaCallback, LearningRateScheduler
from model import Transformer



# model params
n_heads = 8
encoder_layers = decoder_layers = 4
d_model = 64 * n_heads
sequence_len = 100
layer_normalization = True
dropout = True
residual_connections = True

# training params
epochs = 100
batch_size = 30
warmup_steps = 100
optimizer = 'adadelta'
training_data = BEATLES(sequence_len=sequence_len, batch_size=batch_size)
vocab_size = training_data.vocab_size + 1

model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len,
        layer_normalization=layer_normalization, dropout=dropout,
        residual_connections=residual_connections)


def generate_text(epoch, logs, mode='random'):
    # if epoch % 10 > 0:
    #     return
    i = np.random.randint(vocab_size)
    x = np.zeros((1, sequence_len))
    terminate = training_data.char_map[training_data.TERMINATE]
    next_idx = -1
    text = ''.join(training_data.idx_map[i] for i in x[0])
    print('\nusing seed', repr(text))
    while next_idx != terminate and len(text) < 1000:
        pred = model.predict([x, x])
        probs = pred[0][-1]
        next_idx = np.random.choice(range(len(probs)), p=probs)
        text += training_data.idx_map[next_idx]
        # shift elements backward
        x = np.roll(x, -1)
        x[0, -1] = next_idx
    print(repr(text))

def lr_schedule(epoch):
    epoch += 1
    lr = d_model**-.5 * min(epoch**-.5, epoch*warmup_steps**-1.5)
    return lr

callbacks = []
callbacks.append(LambdaCallback(on_epoch_end=generate_text))
callbacks.append(LearningRateScheduler(lr_schedule))

model.summary(line_length=100)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
def gen():
    for i in training_data:
        (x, x), y = i
        if np.isnan(x).any():
            raise Exception('NaNs in input')
        if np.isnan(y).any():
            raise Exception('NaNs in output')
        yield i
gen = gen()
model.fit_generator(gen, steps_per_epoch=len(training_data.files)*10,
                    epochs=epochs, callbacks=callbacks)
model.save('model.h5')

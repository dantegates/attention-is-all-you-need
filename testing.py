import numpy as np
from data import BEATLES
from keras.callbacks import LambdaCallback, LearningRateScheduler
from model import Transformer

n_heads = 8
encoder_layers = decoder_layers = 2
d_model = 64 * n_heads
sequence_len = 10
warmup_steps = 40

training_data = BEATLES(max_len=sequence_len)
x, y = training_data.x, training_data.y
vocab_size = training_data.vocab_size + 1

model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len)


def generate_text(epoch, logs):
    if epoch % 10 > 0:
        return
    i = np.random.randint(len(x))
    xi = x[i].reshape((1, -1))
    terminate = training_data.char_map[training_data.TERMINATE]
    next_idx = -1
    text = ''.join(training_data.idx_map[i] for i in xi[0])
    while next_idx != terminate and len(text) < 100:
        pred = model.predict([xi, xi])
        probs = pred[0][-1]
        next_idx = np.random.choice(range(len(probs)), p=probs)
        text += training_data.idx_map[next_idx]
    print(text)

def lr_schedule(epoch):
    epoch += 1
    lr = d_model**-.5 * min(epoch**-.5, epoch*warmup_steps**-1.5)
    return lr

callbacks = []
callbacks.append(LambdaCallback(on_epoch_end=generate_text))
callbacks.append(LearningRateScheduler(lr_schedule))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit([x, x], y, batch_size=30, epochs=100, callbacks=callbacks)
model.save('model.h5')

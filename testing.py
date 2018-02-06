from keras.callbacks import LambdaCallback
import numpy as np

from model import Transformer
from data import BEATLES


n_heads = 8
encoder_layers = decoder_layers = 8
d_model = 64 * n_heads
sequence_len = 15

training_data = BEATLES(max_len=sequence_len)
x, y = training_data.x, training_data.y
vocab_size = training_data.vocab_size + 1

model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len)


def generate_text(epoch, logs):
    if epoch % 10 > 0:
        return
    xi = x[0].reshape((1, -1))
    terminate = training_data.char_map[training_data.TERMINATE]
    next_idx = -1
    text = ''
    i = 0
    while next_idx != terminate and i < 100:
        pred = model.predict([xi, xi])
        probs = pred[0][-1]
        next_idx = np.random.choice(range(len(probs)), p=probs)
        text += training_data.idx_map[next_idx]
        i += 1
    print(text)

cb = LambdaCallback(on_epoch_end=generate_text)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit([x, x], y, batch_size=30, epochs=100, callbacks=[cb])
model.save('model.h5')

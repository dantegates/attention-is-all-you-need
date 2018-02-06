from model import Transformer
from data import training_data


n_heads = 8
encoder_layers = decoder_layers = 1
d_model = 64 * n_heads
sequence_len = 15

x, y, vocab_size = training_data(sequence_len)
vocab_size = vocab_size + 1

model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit([x, x], y, batch_size=30, epochs=10)
model.save('model.h5')

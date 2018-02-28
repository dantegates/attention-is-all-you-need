import os
from collections import defaultdict, deque

import keras
import numpy as np
from data import BEATLES, CNN, SONGNAMES_TRAIN, SONGNAMES_TEST
from keras.callbacks import (LambdaCallback, LearningRateScheduler,
                             TerminateOnNaN, ModelCheckpoint)
from model import Transformer
loss = 'categorical_crossentropy'

# model params
n_heads = 8
encoder_layers = decoder_layers = 6
d_model = 64 * n_heads
sequence_len = 200
layer_normalization = True
dropout = True
residual_connections = True

# training params
epochs = 250
batch_size = 30
warmup_steps = 1000
optimizer = keras.optimizers.adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
logfile = 'lyrics_train.log'
step_size = 3
tokenizer = 'words'
max_vocab_size = 8000 # redefined later
batch_generator = SONGNAMES_TRAIN(
    sequence_len=sequence_len,
    batch_size=batch_size, step_size=step_size,
    tokenizer=tokenizer, max_vocab_size=max_vocab_size)
batch_generator_test = SONGNAMES_TEST(
    sequence_len=sequence_len,
    batch_size=batch_size, step_size=step_size,
    tokenizer=tokenizer, max_vocab_size=max_vocab_size)
vocab_size = batch_generator.vocab_size + 1

model = Transformer(
        n_heads=n_heads, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        d_model=d_model, vocab_size=vocab_size, sequence_len=sequence_len,
        layer_normalization=layer_normalization, dropout=dropout,
        residual_connections=residual_connections)


def generate_text(epoch, logs, n_beams=3, beam_width=3):
    remove = [batch_generator_test.PAD, batch_generator_test.END, batch_generator_test.UNKOWN]
    token = object()
    encoder_tokens, _, _ = next(batch_generator_test.fetch_examples())
    # copy the tokens!
    encoder_tokens, decoder_tokens = encoder_tokens[:], []
    x1 = batch_generator_test.tokens_to_x(encoder_tokens).reshape((1, -1))
    x2 = batch_generator_test.tokens_to_x(decoder_tokens).reshape((1, -1))
    x = [x1, x2]
    beams = []
    while True:
        # predict and sample an index according to probability dist.
        if not beams:
            pred = model.predict(x)
            probs = np.log10(pred[0][-1])
            indices = np.argsort(probs)[:beam_width]
            for idx in indices:
                token = batch_generator_test.idx_to_char(idx)
                p = probs[idx]
                beams.append(([token], p))
        else:
            new_beams = []
            unfinished = False
            for tokens, prob in beams:
                if tokens[-1] is batch_generator_test.END or \
                        len(tokens) >= sequence_len:
                    continue
                unfinished = True
                x2 = batch_generator_test.tokens_to_x(tokens).reshape((1, -1))
                x = [x1, x2]
                pred = model.predict(x)
                probs = np.log10(pred[0][-1])
                indices = np.argsort(probs)[:beam_width]
                for idx in indices:
                    token = batch_generator_test.idx_to_char(idx)
                    p = probs[idx]
                    new_beams.append((tokens[:] + [token], p*prob))
            beams = new_beams
            if not unfinished:
                break
    tokens = sorted(beams, lambda x: x[1])[-1]
    # remove special tokens
    text = ' '.join(t for t in tokens if not t in remove)
    print('lyrics')
    print(' '.join(encoder_tokens))
    print('generated song title:', text)


def lr_schedule(epoch):
    epoch += 1
    lr = d_model**-.5 * min(epoch**-.5, epoch*warmup_steps**-1.5)
    return lr

callbacks = []
callbacks.append(LambdaCallback(on_epoch_end=generate_text))
callbacks.append(LearningRateScheduler(lr_schedule))
callbacks.append(TerminateOnNaN())
callbacks.append(ModelCheckpoint(filepath='lyrics_model.h5', period=1, save_weights_only=True))

# for debugging. e.g. if loss turns to NaN, batches[0] will contain batch
# that caused the NaN
batches = deque(maxlen=2)
gen = (i for i in batch_generator)
gen_test = (i for i in batch_generator_test)

#from keras import backend as K
#def loss(y_true, y_pred):
 #   return K.categorical_crossentropy(y_true[:,-20:,:], y_pred[:,-20:,:])


if __name__ == '__main__':
    if os.path.exists(logfile):
        # clear log file
        with open(logfile, 'w'):
            pass
    model.summary()
    model.compile(loss=loss, optimizer=optimizer)
    try:
        model.fit_generator(gen, steps_per_epoch=batch_generator.n_batches,
                            epochs=epochs, callbacks=callbacks,
                            validation_data=gen_test,
                            validation_steps=batch_generator_test.n_batches)
    except KeyboardInterrupt:
        pass
model.save('lyrics_model.h5')

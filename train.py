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
encoder_layers = decoder_layers = 2
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
logfile = 'train-predicitons.txt'
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


def beam_predict(model, x1, x2, fan_out, beam_width, terminal, max_len):
    def predict(X, n):
        preds = model.predict(X)
        # remove batch-shape, take prediction for last item in sequence
        preds = preds[0][-1]
        if n > 1:
            indices = np.argsort(preds)[:n]
            probs = np.log10(preds[indices])
            return zip(indices, probs)
        idx = np.argmax(preds)
        return idx

    beams = []
    x = [x1, x2]
    # beam search to find most likely prediction
    for _ in range(beam_width):
        # predict and sample an index according to probability dist.
        if not beams:
            for idx, p in predict(x, fan_out):
                x2 = np.append(x2.copy(), [idx])
                beams.append((x2, p))
        else:
            new_beams = []
            for x2, prob in beams:
                x = [x1, x2]
                for idx, p in predict(x, fan_out):
                    x2 = np.append(x2.copy(), [idx])
                    new_beams.append((x2, p*prob))
            beams = new_beams
    # take top prediciton
    x2, _ = sorted(beams, key=lambda x: x[1])[-1]

    # generate the rest of the text given the most likely beam
    while x2[-1] is not terminal and len(x2) <= max_len:
        x = [x1, x2]
        idx = predict(x, 1)
        x2 = np.append(x2, [idx])

    return x2


def generate_text(epoch, logs, fan_out=3, beam_width=3):
    # pick a random example to seed predictions
    # make sure to copy the tokens!
    predictions = []
    for i in range(10):
        decoder_tokens = []
        encoder_tokens, _, _ = next(batch_generator_test.fetch_examples())[:]# copy the tokens!

        # make input for model
        x1 = batch_generator_test.tokens_to_x(encoder_tokens).reshape((1, -1))
        x2 = batch_generator_test.tokens_to_x(decoder_tokens).reshape((1, -1))

        pred = beam_predict(model, x1, x2, fan_out, beam_width, batch_generator_test.END,
                            sequence_len)
        tokens = [batch_generator_test.idx_to_char(idx) for idx in pred]

        # format generated text
        remove = [batch_generator_test.PAD, batch_generator_test.END, batch_generator_test.UNKOWN]
        text = ' '.join(t for t in tokens if not t in remove)
        predictions.append({'seed': encoder_tokens, 'generated_text': text})
    json_data = {
        'epoch': epoch,
        'logs': logs,
        'predictions': predictions}
    with open(logfile, 'a') as f:
        f.write(json.dumps(json_data))

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

# attention-is-all-you-need
keras implementation of attention is all you need

# Python files

- [model.py](./model.py) Implements the Transformer model.

- [model_decoder.py](./model_decoder.py) Implements a decoder only variation of the Transformer used for text generation. The Transformer in [model.py](./model.py) has mostly incorporated this feature and eventually this file will be removed.

- [data.py](./data.py) A generic batch generator useful for working with keras. Its a lenghty implementation so it lives in a Python file and is subclassed with the Attention Is All You Need specific details in the [training notebook](./notebooks.train.ipynb).

- [summary_batch_generator.py](./summary_batch_generator.py) This class is implemented in the [training notebook](./notebooks.train.ipynb). Eventually this file will be removed.

- [test.py](./test.py) Unit tests. Run with `python test.py`.

# notebooks

- [preprocess-stories](./notebooks/preprocess-stories.ipynb) Builds the summary data set. I.e. Writes to disk the story with summary built from higlights.

- [byte-pair-encoding](./notebooks/byte-pair-encoding.ipynb) Interactive implementation of nicely wrapping the sentencepiece BPE API.

- [train](./notebooks/train.ipynb) Train the Transformer model.

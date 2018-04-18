from keras.preprocessing.text import text_to_word_sequence, Tokenizer as _Tokenizer


# extend keras tokenizer class to
#   1. reserve the tokens 0 (pad), 1 (input/target separator), 2 (end of target)
#      and 3 (out of vocab)
#   2. respect num_words when tokenizing


class Tokenizer(_Tokenizer):
    def fit_on_texts(self, texts):
        """Updates internal vocabulary based on a list of texts.
        In the case where texts contains lists, we assume each entry of the lists
        to be a token.
        Required before using `texts_to_sequences` or `texts_to_matrix`.
        # Arguments
            texts: can be a list of strings,
                a generator of strings (for memory-efficiency),
                or a list of list of strings.
        """
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0, 1, 2 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(4, len(sorted_voc) + 4)))))
        self.word_index[self.oov_token] = 3

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences_generator(self, texts):
        """Transforms each text in `texts` in a sequence of integers.
        Each item in texts can also be a list, in which case we assume each item of that list
        to be a token.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Yields
            Yields individual sequences.
        """
        num_words = self.num_words
        for text in texts:
            if self.char_level or isinstance(text, list):
                seq = text
            else:
                seq = text_to_word_sequence(text,
                                            self.filters,
                                            self.lower,
                                            self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None and (self.num_words and i < self.num_words):
                    vect.append(i)
                elif self.oov_token is not None:
                    i = self.word_index.get(self.oov_token)
                    if i is not None:
                        vect.append(i)
            yield vect
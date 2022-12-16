import np

from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

class AverageEmbeddingVectorizer():

    def __init__(self, word_index=None):
        self._word_index = word_index

    def fit (self, X, y):
        return self

    def fit_transform (self, X, y):
        return self.fit(X, y).transform(X)

    def transform(self, X):
        sentences = [word_tokenize(row) for row in X]

        result = []
        for sentence in sentences:
            vec = None
            n_words = 0
            for word in sentence:
                if word in self._word_index:
                    if vec is None:
                        vec = self._word_index[word]
                    else:
                        vec = vec + self._word_index[word]
                    n_words = n_words + 1

            if vec is None:
                result.append(np.zeros(
                    list(self._word_index.items())[0][1].shape))
            else:
                result.append(vec / n_words)

        return np.array(result)

    def get_params (self, deep=False):
        return { 'loader': self._loader }

    def set_params(self, **parameters):
      for parameter, value in parameters.items():
          setattr(self, parameter, value)
      return self


class GloveLoader:

    def __init__ (self, filename):
        self._filename = filename

    def build_word_index (self):
        word_index = {}
        with open(self._filename) as f:
            for line in f:
                word, *vector = line.split()
                word_index[word] = np.array(
                        vector, dtype=np.float32)
        return word_index


class SELoader:

    def __init__ (self, filename):
        self._filename = filename

    def build_word_index (self):
        word_index = {}
        word_vecs = KeyedVectors.load_word2vec_format(
                self._filename, binary=True)

        return word_vecs

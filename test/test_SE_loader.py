from unittest import TestCase
from unittest.mock import Mock, patch

from util.embedding_vectorizer import SELoader

class SELoaderTest (TestCase):

    def test_loads_embeddings (self):
        loader = SELoader('../SE_embeddings.bin')
        wv_mock = Mock()

        with patch('gensim.models.KeyedVectors.load_word2vec_format') as mock:
            mock.return_value = wv_mock
            result = loader.build_word_index()
            self.assertEqual(wv_mock, result)
            mock.assert_called_once_with(
                    '../SE_embeddings.bin', binary=True)


    def test_loads_embeddings_and_retrieves_vector (self):
        loader = SELoader('../SE_embeddings2.bin')
        wv_mock = Mock()
        wv_mock.get_vector.return_value = [1, 2, 3]

        with patch('gensim.models.KeyedVectors.load_word2vec_format') as mock:
            mock.return_value = wv_mock
            result = loader.build_word_index()

            mock.assert_called_once_with(
                    '../SE_embeddings2.bin', binary=True)
            self.assertEqual([1, 2, 3], result.get_vector('abobrinha'))
            wv_mock.get_vector.assert_called_once_with('abobrinha')


    def test_retrieves_vector_but_ignores_inexistent_words (self):
        loader = SELoader('../SE_embeddings2.bin')
        wv_mock = Mock()
        fake_word_index = {
                'pepino': [3, 2, 1], 'beterraba': [6, 5, 4] }
        def get_vector_side_effect (word):
            return fake_word_index[word]
        wv_mock.get_vector.side_effect = get_vector_side_effect

        with patch('gensim.models.KeyedVectors.load_word2vec_format') as mock:
            mock.return_value = wv_mock
            result = loader.build_word_index()
            mock.assert_called_once_with(
                    '../SE_embeddings2.bin', binary=True)
            self.assertEqual([3, 2, 1], result.get_vector('pepino'))
            self.assertEqual([6, 5, 4], result.get_vector('beterraba'))
            wv_mock.get_vector.assert_any_call('pepino')
            wv_mock.get_vector.assert_any_call('beterraba')


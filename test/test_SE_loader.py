from unittest import TestCase
from unittest.mock import Mock, patch

from util.embedding_vectorizer import SELoader

class SELoaderTest (TestCase):

    def test_loads_embeddings (self):
        word_list = set([])
        loader = SELoader('../SE_embeddings.bin')
        wv_mock = Mock()

        with patch('gensim.models.KeyedVectors.load_word2vec_format') as mock:
            mock.return_value = wv_mock
            result = loader.build_word_index(word_list)
            self.assertEqual({}, result)
            mock.assert_called_once_with(
                    '../SE_embeddings.bin', binary=True)

    def test_loads_embeddings_and_retrieves_vector (self):
        word_list = set(['abobrinha'])
        loader = SELoader('../SE_embeddings2.bin')
        wv_mock = Mock()
        wv_mock.get_vector.return_value = [1, 2, 3]

        with patch('gensim.models.KeyedVectors.load_word2vec_format') as mock:
            mock.return_value = wv_mock
            result = loader.build_word_index(word_list)
            mock.assert_called_once_with(
                    '../SE_embeddings2.bin', binary=True)
            wv_mock.get_vector.assert_called_once_with('abobrinha')
            self.assertIn('abobrinha', result)
            self.assertEqual(
                    [1, 2, 3], result['abobrinha'].tolist())


    def test_retrieves_vector_but_ignores_inexistent_words (self):
        word_list = set(['pepino', 'limao', 'beterraba'])
        loader = SELoader('../SE_embeddings2.bin')
        wv_mock = Mock()
        fake_word_index = {
                'pepino': [3, 2, 1], 'beterraba': [6, 5, 4] }
        def get_vector_side_effect (word):
            return fake_word_index[word]
        wv_mock.get_vector.side_effect = get_vector_side_effect

        with patch('gensim.models.KeyedVectors.load_word2vec_format') as mock:
            mock.return_value = wv_mock
            result = loader.build_word_index(word_list)
            mock.assert_called_once_with(
                    '../SE_embeddings2.bin', binary=True)
            wv_mock.get_vector.assert_any_call('pepino')
            wv_mock.get_vector.assert_any_call('limao')
            wv_mock.get_vector.assert_any_call('beterraba')
            self.assertIn('pepino', result)
            self.assertEqual(
                    [3, 2, 1], result['pepino'].tolist())
            self.assertIn('beterraba', result)
            self.assertEqual(
                    [6, 5, 4], result['beterraba'].tolist())


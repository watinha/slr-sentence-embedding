from unittest import TestCase
from unittest.mock import Mock, patch, mock_open

from util.embedding_vectorizer import AverageEmbeddingVectorizer, GloveLoader

class AverageEmbeddingVectorizerTest (TestCase):

    def test_vectorize_sentence (self):
        text = 'abobrinha pepino mamao'
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.txt'))
        mock = mock_open()
        mock.return_value.__iter__ = Mock(return_value = iter([
            'abobrinha 1 2 3',
            'pepino 4 5 6',
            'mamao 7 8 9']))

        with patch('util.embedding_vectorizer.open', mock):
            result = vectorizer.transform([text])

        mock.assert_called_with('glove.txt')
        self.assertEqual([[4, 5, 6]], result.tolist())

    def test_vectorize_different_sentence (self):
        text = 'legal ultra'
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.2d.txt'))
        mock = mock_open()
        mock.return_value.__iter__ = Mock(return_value = iter([
            'legal 1 2', 'ultra 4 5']))

        with patch('util.embedding_vectorizer.open', mock):
            result = vectorizer.transform([text])

        mock.assert_called_with('glove.2d.txt')
        self.assertEqual([[2.5, 3.5]], result.tolist())

    def test_vectorize_should_ignore_unused_words (self):
        text = 'legal ultra'
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.2d.txt'))
        mock = mock_open()
        mock.return_value.__iter__ = Mock(return_value = iter([
            'legal 1 2', 'nothing 12 13', 'ultra 4 5']))

        with patch('util.embedding_vectorizer.open', mock):
            result = vectorizer.transform([text])

        mock.assert_called_with('glove.2d.txt')
        self.assertEqual([[2.5, 3.5]], result.tolist())


    def test_vectorize_should_run_on_multiple_sentences (self):
        corpus = ['nova alternativa de jogo',
                  'terceira alternativa legal',
                  'outra sentenca']
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.2d.txt'))
        mock = mock_open()
        mock.return_value.__iter__ = Mock(return_value = iter([
            'nova 3 9', 'alternativa 13 11', 'nada 3 10',
            'de 9 1', 'jogo 13 1', 'terceira 4 11',
            'outra 13 9', 'sentenca 1 1',
            'legal 1 2', 'nothing 12 13', 'ultra 4 5']))

        with patch('util.embedding_vectorizer.open', mock):
            result = vectorizer.transform(corpus)

        mock.assert_called_with('glove.2d.txt')
        self.assertAlmostEqual([
            [9.5, 5.5],
            [6, 8],
            [7, 5]], result.tolist())


    def test_vectorize_should_ignore_words_not_in_embeddings (self):
        corpus = ['nova alternativa de jogo',
                  'alternativa terceira alternativa legal',
                  'outra sentenca']
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.2d.txt'))
        mock = mock_open()
        mock.return_value.__iter__ = Mock(return_value = iter([
            'nova 3 9', 'nada 3 10', 'de 9 1',
            'jogo 12 2', 'terceira 4 11',
            'outra 13 9', 'sentenca 1 1',
            'legal 1 2', 'nothing 12 13', 'ultra 4 5']))

        with patch('util.embedding_vectorizer.open', mock):
            result = vectorizer.transform(corpus)

        mock.assert_called_with('glove.2d.txt')
        self.assertEqual([
            [8.0, 4.0],
            [2.5, 6.5],
            [7.0, 5.0]], result.tolist())

    def test_sentences_with_no_word_in_word_index (self):
        corpus = ['nova alternativa de jogo',
                  'alternativa terceira alternativa legal',
                  'natal sem nenhuma palavra',
                  'outra sentenca']
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.2d.txt'))
        mock = mock_open()
        mock.return_value.__iter__ = Mock(return_value = iter([
            'nova 3 9', 'nada 3 10', 'de 9 1',
            'jogo 12 2', 'terceira 4 11',
            'outra 13 9', 'sentenca 1 1',
            'legal 1 2', 'nothing 12 13', 'ultra 4 5']))

        with patch('util.embedding_vectorizer.open', mock):
            result = vectorizer.transform(corpus)

        mock.assert_called_with('glove.2d.txt')
        self.assertEqual([
            [8.0, 4.0],
            [2.5, 6.5],
            [0.0, 0.0],
            [7.0, 5.0]], result.tolist())

    def test_sentences_with_no_word_in_index_with_3d (self):
        corpus = ['natal sem nenhuma palavra',
                  'outra sentenca']
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.3d.txt'))
        mock = mock_open()
        mock.return_value.__iter__ = Mock(return_value = iter([
            'outra 1 1 1', 'sentenca 1 1 1', 'ultra 4 5 6']))

        with patch('util.embedding_vectorizer.open', mock):
            result = vectorizer.transform(corpus)

        mock.assert_called_with('glove.3d.txt')
        self.assertEqual([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]], result.tolist())

    def test_vectorize_implements_fit_interface (self):
        X_stub, y_stub = [], []
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.2d.txt'))
        result = vectorizer.fit(X_stub, y_stub)
        self.assertEqual(vectorizer, result)

    def test_vectorize_implements_fit_transform_interface (self):
        corpus = ['nova alternativa de jogo',
                  'alternativa terceira alternativa legal',
                  'outra sentenca']
        vectorizer = AverageEmbeddingVectorizer(
                GloveLoader('glove.2d.txt'))
        vectorizer.fit = Mock(return_value=vectorizer)
        mock = mock_open()
        mock.return_value.__iter__ = Mock(return_value = iter([
            'nova 3 9', 'nada 3 10', 'de 9 1',
            'jogo 12 2', 'terceira 4 11',
            'outra 13 9', 'sentenca 1 1',
            'legal 1 2', 'nothing 12 13', 'ultra 4 5']))
        y_stub = []

        with patch('util.embedding_vectorizer.open', mock):
            result = vectorizer.fit_transform(corpus, y_stub)

        mock.assert_called_with('glove.2d.txt')
        self.assertEqual([
            [8.0, 4.0],
            [2.5, 6.5],
            [7.0, 5.0]], result.tolist())
        vectorizer.fit.assert_called_once_with(corpus, y_stub)



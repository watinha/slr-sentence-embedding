from sklearn import tree, svm, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from gensim.models.keyedvectors import load_word2vec_format

from util.embedding_vectorizer import AverageEmbeddingVectorizer
from util.text_filter import StopwordsFilter, LemmatizerFilter


slrs_files = {
    'games': {
        'argument': [ 'bibs/games/round1-todos.bib' ],
        'project_folder': 'games'
    },
    'slr': {
        'argument': [ 'bibs/slr/round1-todos.bib' ],
        'project_folder': 'slr'
    },
    'pair': {
        'argument': [ 'bibs/pair/round1-todos.bib' ],
        'project_folder': 'pair'
    },
    'illiterate': {
       'argument': [ 'bibs/illiterate/round1-others.bib' ],
       'project_folder': 'illiterate'
    },
    'mdwe':{
       'argument': [ 'bibs/mdwe/round1-acm.bib',
           'bibs/mdwe/round1-ieee.bib', 'bibs/mdwe/round1-sciencedirect.bib' ],
       'project_folder': 'mdwe'
    },
    'testing': {
       'argument': [ 'bibs/testing/round1-google.bib',
           'bibs/testing/round1-ieee.bib', 'bibs/testing/round1-outros.bib',
           'bibs/testing/round2-google.bib', 'bibs/testing/round2-ieee.bib',
           'bibs/testing/round2-outros.bib', 'bibs/testing/round3-google.bib'],
       'project_folder': 'testing'
    },
    'ontologies': {
       'argument': [ 'bibs/ontologies/round1-google.bib',
           'bibs/ontologies/round1-ieee.bib', 'bibs/ontologies/round1-outros.bib',
           'bibs/ontologies/round2-google.bib', 'bibs/ontologies/round2-ieee.bib',
           'bibs/ontologies/round3-google.bib' ],
       'project_folder': 'ontologies'
    },
    'xbi': {
       'argument': [ 'bibs/xbi/round1-google.bib',
           'bibs/xbi/round1-ieee.bib', 'bibs/xbi/round1-outros.bib',
           'bibs/xbi/round2-google.bib', 'bibs/xbi/round2-ieee.bib',
           'bibs/xbi/round3-google.bib' ],
       'project_folder': 'xbis'
   }
}

def get_slr_files(slr):
    return slrs_files[slr]['argument']

seed = 42
def get_classifier(classifier_name):
    classifier = None
    params = {}
    if (classifier_name == 'svm'):
        classifier = svm.SVC(random_state=seed, probability=True)
        params = {
            #'kernel': ['linear', 'rbf', 'poly'],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__C': [1, 10, 100],
            #'degree': [2, 3],
            'classifier__coef0': [0, 10, 100],
            'classifier__tol': [0.001, 0.1, 1],
            'classifier__class_weight': ['balanced', None]
        }
    elif (classifier_name == 'dt'):
        classifier = tree.DecisionTreeClassifier(random_state=seed)
        params = {
            'classifier__criterion': ["gini", "entropy"],
            'classifier__max_depth': [5, 10, 50, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__class_weight': [None, 'balanced']
        }
    elif (classifier_name == 'rf'):
        classifier = ensemble.RandomForestClassifier(random_state=seed)
        params = {
            'classifier__n_estimators': [5, 10, 100],
            'classifier__criterion': ["gini", "entropy"],
            'classifier__max_depth': [10, 50, 100, None],
            'classifier__min_samples_split': [2, 10, 100],
            'classifier__class_weight': [None, 'balanced']
        }
    else:
        classifier = svm.LinearSVC(random_state=seed, fit_intercept=False)
        params = {
            'classifier__C': [1, 10, 100],
            'classifier__tol': [0.001, 0.1, 1],
            'classifier__class_weight': ['balanced', None]
        }
    return classifier, params


def get_selector (selector_name):
    if selector_name == 'selectkbest':
      return SelectKBest(chi2), { 'selector__k': [ 25, 50, 100, 200, 'all'] }
    if selector_name == 'truncatedsvd':
      return TruncatedSVD(), { 'selector__n_components': [ 25, 50, 100, 200] }


def get_filters(extractor_name):
  if extractor_name == 'tfidf':
    return [StopwordsFilter(), LemmatizerFilter()]
  else:
    return [StopwordsFilter()]


cache = {}
def get_extractor(extractor_name, embeddings_filename=''):
    if extractor_name == 'tfidf':
        return TfidfVectorizer(ngram_range=(1,3)), StandardScaler(with_mean=False)
    elif extractor_name == 'embeddings_glove':
      if extractor_name not in cache:
        print('     - building word index: %s' % (embeddings_filename))
        cache[extractor_name] = load_word2vec_format(embeddings_filename, no_header=True)

      return AverageEmbeddingVectorizer(cache[extractor_name]), MinMaxScaler()
    elif extractor_name == 'embeddings_se':
      if extractor_name not in cache:
        print('     - building word index: %s' % (embeddings_filename))
        cache[extractor_name] = load_word2vec_format(embeddings_filename, binary=True)

      return AverageEmbeddingVectorizer(cache[extractor_name]), MinMaxScaler()



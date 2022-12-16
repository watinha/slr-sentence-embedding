import np, sys

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from config import get_slr_files, get_classifier, get_extractor, get_selector, get_filters
from util.bib_loader import load
from util.years_split import YearsSplit
from util.text_filter import FilterComposite


if (len(sys.argv) < 2):
    print('second argument missing: SLR theme (games|slr|illiterate|pair|mdwe|testing|ontologies|xbi)')
    sys.exit(1)

if (len(sys.argv) < 3):
    print('third argument missing: classifier (svm|dt|rf|lsvm)')
    sys.exit(1)

if (len(sys.argv) < 4):
    print('forth argument missing: titles only?')
    sys.exit(1)

if (len(sys.argv) < 5):
    print('fifth argument missing: extrator (tfidf,embeddings_glove,embeddings_se)')
    sys.exit(1)

if (len(sys.argv) < 6):
    print('sixth argument missing: selector (selectkbest, truncatedsvd)')
    sys.exit(1)

_, theme, classifier_name, titles, extractor_name, selector_name = sys.argv
titles = True if titles == 'true' else False
embedding_filename = './embeddings/glove.6B.200d.txt' if extractor_name == 'embeddings_glove' else './embeddings/SO_vectors_200.bin'

print('- Loading SLR data: %s' % (theme))
slr_files = get_slr_files(theme)
X, y, years = load(slr_files, titles_only=titles)

y_pred = []
y_true = []
y_pred_prob = []
excluded = []
missed = []

X = np.array(X)
y = np.array(y)


print('- Preprocessing data (removing stopwords and lemmatizer)')
filters = get_filters(extractor_name)
preprocessor = FilterComposite(filters=filters)
X = np.array(preprocessor.fit_transform(X))

print('- Running Cross-Validation with %s' % (classifier_name))
kfold = YearsSplit(n_split=3, years=years)
for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print('   - Extracting features run...')
    extractor, scaler = get_extractor(extractor_name, embeddings_filename=embedding_filename)
    extractor_pipeline = Pipeline([
      ('extractor', extractor),
      ('scaler', scaler)
    ])
    X_train_ext = extractor_pipeline.fit_transform(X_train)
    X_test_ext = extractor_pipeline.transform(X_test)

    print('   - Grid Search run...')
    classifier, classifier_params = get_classifier(classifier_name)
    selector, selector_params = get_selector(selector_name)

    pipeline = GridSearchCV(Pipeline([
        ('selector', selector),
        ('classifier', classifier)
    ]), { **classifier_params, **selector_params }, cv=2, scoring='roc_auc')

    pipeline.fit(X_train_ext, y_train)
    y_pred = y_pred + (pipeline.predict(X_test_ext)).tolist()
    y_true = y_true + y_test.tolist()

    true_class_index = list(pipeline.classes_).index(1)
    y_prob = pipeline.predict_proba(X_train_ext)[:, true_class_index]
    _, _, thresholds = metrics.precision_recall_curve(y_train, y_prob)
    threshold = min(thresholds[0], 0.5)
    y_test_prob = (pipeline.predict_proba(X_test_ext)[:, true_class_index])
    y_test_prob = [ 0 if prob < threshold else 1 for prob in y_test_prob.tolist() ]
    y_pred_prob = y_pred_prob + y_test_prob

    matrix = metrics.confusion_matrix(y_test, y_test_prob)
    excluded.append(
      matrix[0, 0] /
      (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0]))
    missed.append(matrix[1, 0] / (matrix[1, 1] + matrix[1, 0]))


print('')
print('')
print('====================================')
print('=        General accuracy          =')
print('====================================')
print(metrics.classification_report(y_true, y_pred))
print('            ROC-AUC: %f' % (metrics.roc_auc_score(y_true, y_pred)))
print('====================================')
print('Configuring the activation threshold')
print('====================================')
print(metrics.classification_report(y_true, y_pred_prob))
print('            ROC-AUC: %f' % (metrics.roc_auc_score(y_true, y_pred_prob)))
print('====================================')
print('         SLR Updates metrics')
print('====================================')
print('Excluded percentage: %f' % (sum(excluded) / len(excluded)))
print('  Missed percentage: %f' % (sum(missed) / len(missed)))


sys.exit(0)

import numpy as np
import pandas as pd

from preprocessing import load_preprocessed_data

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


X_data, y_data = load_preprocessed_data('train')
test_data = load_preprocessed_data('test')
submit_data = pd.read_csv('data/gender_submission.csv')

sgd_clf = SGDClassifier(penalty='none')
rasso_sgd = SGDClassifier(penalty='l1')
ridge_sgd = SGDClassifier(penalty='l2')
elastic_sgd = SGDClassifier(penalty='elasticnet')
log_clf = LogisticRegression(penalty='none')
ridge_log = LogisticRegression() # ridge regulization is default
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

models = {
    '0': sgd_clf, 
    '1': rasso_sgd, 
    '2': ridge_sgd, 
    '3': elastic_sgd, 
    '4': log_clf, 
    '5': ridge_log,
    '6': rnd_clf,
    '7': svm_clf
    }

voting_clf = VotingClassifier(
    estimators=[(key, model) for key, model in models.items()],
    voting='hard'
)
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm='SAMME.R', learning_rate=0.5
)

models.update({
    '8': voting_clf,
    '9': ada_clf
    })

skfolds = StratifiedKFold(n_splits=5, shuffle=False)

for clf in [sgd_clf]:
    scores = []
    for train_index, test_index in skfolds.split(X_data, y_data):
        X_train_folds = X_data[train_index]
        y_train_folds = y_data[train_index]
        X_test_fold = X_data[test_index]
        y_test_fold = y_data[test_index]

        clf.fit(X_train_folds, y_train_folds)
        y_pred = clf.predict(X_test_fold)
        score = sum(y_pred == y_test_fold) / len(y_pred)
        scores.append(score)

    print('clf:', clf, ', scores:', scores)

    y_pred = clf.predict(X_data)
    print('confusion matrix:\n', confusion_matrix(y_data, y_pred))
    print('precision_score:', precision_score(y_data, y_pred), \
    ', recall score:', recall_score(y_data, y_pred), ', f1 score:', \
        f1_score(y_data, y_pred), '\n')

    prediction = clf.predict(test_data)

    submission = pd.DataFrame({
        'PassengerId': submit_data['PassengerId'],
        'Survived': prediction
    })

    if clf == voting_clf:
        name = 'VotingClassifier'
    elif clf == ada_clf:
        name = 'AdaBoostClassifier'
    else:
        name = clf

    submission.to_csv(f'data/{name}_submission.csv', index=False)
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import xgboost as xgb
import numpy as np


class ClassifierRandomForest(BaseEstimator):
    def __init__(self):
        print("I don't like classifying people :(")

    def fit(self, X, y):
        self.clf = RandomForestClassifier(
            n_estimators=2, max_leaf_nodes=2, random_state=61)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


# ----------------------------------------------------------------------

class ClassifierSimpleTree(BaseEstimator):
    def __init__(self, nleaf=20):
        self.clf = DecisionTreeClassifier(
            max_depth=20,
            max_features=57,
            max_leaf_nodes=nleaf
        )
        self.nleaf = nleaf
        self.feats = nleaf
        self.name = 'ClassifierSimpleTree' + '_nleaf' + str(self.nleaf)
        print(self.name)

    def fit(self, X, y):
        self.clf.fit(X, y)
        with open("simple_tree.dot", 'w') as f:
            f = export_graphviz(self.clf, out_file=f)

        importances = self.clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" %
                  (f, indices[f], importances[indices[f]]))

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class ClassifierXGB(BaseEstimator):
    def __init__(self, level=0):
        self.name = 'ClassifierXGB' + '_feat' + str(level)
        print(self.name)
        pass

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        param = {'objective': 'binary:logistic', 'max_depth': 8,
                 'subsample': 1, 'eta': 0.1, 'num_round': 400,
                 'gamma': 0, 'silent': 1}
        self.clf = xgb.train(param, dtrain,
                             num_boost_round=param['num_round'])
        print("weight scores")
        print(self.clf.get_score(importance_type="weight"))
        print("gain scores")
        print(self.clf.get_score(importance_type="gain"))
        print("cover scores")
        print(self.clf.get_score(importance_type="cover"))

    def predict(self, X):
        dpred = xgb.DMatrix(X)
        return (np.sign(self.clf.predict(dpred) - 0.5) / 2 + 0.5)

    def predict_proba(self, X):
        dpred = xgb.DMatrix(X)
        pred = self.clf.predict(dpred)
        pred = np.array([1 - pred, pred]).T
        return pred


Classifier = ClassifierSimpleTree
# Classifier = ClassifierXGB

# class Classifier(ClassifierSimpleTree):
#    def __init__(self):
#        ClassifierSimpleTree.__init__(self)
#        pass

import numpy as np
from submissions.formulas.formulate import formulate


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        print("I'm feature extractor and I'm fitting...")
        X = self.transform(X_df)
        print(X[-1:5, :])

    def transform(self, X_df):
        print("I'm transforming.")
        X = X_df.values

# add column:

#       formula = formulate()
        formula = "X[:, 3] * X[:, 5]"
        XF = eval(formula)
        X_new = np.concatenate((X, XF.reshape(-1, 1)), axis=1)

# remove column:

        removed_column = 0
        X_new = np.delete(X_new, removed_column, axis=1)
        return X_new

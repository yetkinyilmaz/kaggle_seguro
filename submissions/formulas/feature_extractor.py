import numpy as np
from submissions.formulas.GenerateTrees import generateSingleTree


class FeatureExtractor():
    def __init__(self):
        self.tree = generateSingleTree(5)
        self.n_fit = 10000
        pass

    def remove_column(self, X, removed_column=0):
        X_new = np.delete(X, removed_column, axis=1)
        return X_new

    def fit(self, X_df, y):
        print("I'm feature extractor and I'm fitting...")
        X = self.remove_column(X_df.head(self.n_fit).values, 0)
        y = y[0:self.n_fit]
        self.tree.fit_coefficients(X, y)

    def transform(self, X_df):
        print("I'm transforming.")
# remove index column
        X = self.remove_column(X_df.values, 0)

# add column:
        C = self.tree.coefficients
        XF = eval(self.tree.print_tree())
        X_new = np.concatenate((X, XF.reshape(-1, 1)), axis=1)

# remove non-trivial column:
        X_new = self.remove_column(X_new, 0)
        return X_new

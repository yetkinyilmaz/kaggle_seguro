import numpy as np


class FeatureExtractor():
    def __init__(self):
        pass

    def remove_column(self, X, removed_column=0):
        X_new = np.delete(X, removed_column, axis=1)
        return X_new

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X = self.remove_column(X_df.values, 0)
        return X

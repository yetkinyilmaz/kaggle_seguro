import numpy as np
from submissions.formulas.formulate import formulate


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        print("I'm feature extractor and I'm fitting...")
        X = self.transform(X_df)
        print(X[-1:5,:])


    def transform(self, X_df):
        print("I'm transforming.")
        X = X_df.values
 #       formula = formulate()
        feature = []
        feature.append((X[:,3] * X[:,5]).reshape(-1, 1))
        print(feature)
        X_new = np.concatenate((X, np.concatenate(feature, axis=1)), axis=1)
        return X_new[:,1:-1]

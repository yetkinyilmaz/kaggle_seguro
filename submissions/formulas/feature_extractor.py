class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        print("I'm feature extractor and I'm fitting...")

    def transform(self, X_df):
        print("I'm transforming.")
        return X_df.values

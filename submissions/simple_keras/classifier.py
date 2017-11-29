import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.base import BaseEstimator


def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=57))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.clf = create_model()
        self.clf.fit(X, y, epochs=2)
        print(self.predict(X))

    def predict(self, X):
        y = self.clf.predict(X).flatten()
        print(y)
        return y

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

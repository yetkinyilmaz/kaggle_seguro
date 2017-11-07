import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Kaggle Porto-Seguro safe driver prediction'
_target_column_name = 'target'
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    rw.score_types.NormalizedGini(name='ngini', precision=3),
    rw.score_types.ROCAUC(name='auc', precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=57)
    return cv.split(X, y)


def get_train_data(path='.'):
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        data = pd.read_csv(os.path.join(path, 'chopped_data', 'train.csv'))
    else:
        data = pd.read_csv(os.path.join(path, 'kaggle_data', 'train.csv'))
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_test_data(path='.'):
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        X_df = pd.read_csv(os.path.join(path, 'chopped_data', 'test.csv'))
    else:
        X_df = pd.read_csv(os.path.join(path, 'kaggle_data', 'test.csv'))
    y_array = np.zeros(len(X_df))  # dummy labels for syntax
    y_array[0] = 1  # to make AUC work
    return X_df, y_array


def save_y_pred(y_pred, data_path='.', output_path='.', suffix='test'):
    if 'test' not in suffix:
        return  # we don't care about saving the training predictions
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        sample_df = pd.read_csv(os.path.join(
            data_path, 'data', 'sample_submission.csv'))
    else:
        sample_df = pd.read_csv(os.path.join(
            data_path, 'kaggle_data', 'sample_submission.csv'))
    df = pd.DataFrame()
    df['id'] = sample_df['id']
    df['target'] = y_pred[:, 1]
    output_f_name = os.path.join(output_path, 'y_pred_{}.csv'.format(suffix))
    df.to_csv(output_f_name, index=False)

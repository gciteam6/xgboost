# Third-party modules
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from sklearn.externals.joblib import Parallel, delayed
# Hand-made modules
from .base import MyEstimatorBase, PickleMixin, safe_split

NUM_FOLDING = 10
MACHINE_LEARNING_TECHNIQUE_NAME = "blending"
PICKLE_EXTENSION = "pkl"
SERIALIZE_FILENAME_PREFIX = "fit_model"


class MyBlender(MyEstimatorBase, PickleMixin):
    def __init__(self, blend_model_instance: BaseEstimator, model_name: str, params: dict):
        super().__init__(model_name, params)
        self.regr = blend_model_instance.set_params(**params)
        self.MODELS_SERIALIZING_BASEPATH = self.path.join(self.MODELS_SERIALIZING_BASEPATH,
                                                          MACHINE_LEARNING_TECHNIQUE_NAME)
        self.SERIALIZE_FILENAME_PREFIX = SERIALIZE_FILENAME_PREFIX

    def get_model_instance(self):
        return self.regr

    def fit(self, X, y):
        X, y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = \
            _center_scale_xy(X, y)

        fit_model = self.regr.fit(X, y.reshape(-1))
        self.to_pkl(self, self.gen_serialize_filepath(self.SERIALIZE_FILENAME_PREFIX, PICKLE_EXTENSION))

        return fit_model

    def predict(self, X):
        obj = self.read_pkl(self.gen_serialize_filepath(self.SERIALIZE_FILENAME_PREFIX, PICKLE_EXTENSION))

        X = (X - obj.x_mean_) / obj.x_std_
        y_pred = obj.regr.predict(X)

        return (y_pred * obj.y_std_) + obj.y_mean_

    def cross_val_predict(self, X, y, cv=KFold(n_splits=NUM_FOLDING)):
        """From 'cross_val_predict' in sklearn.model_selection._validation"""
        prediction_blocks = Parallel(n_jobs=-1)(delayed(_fit_and_predict)(
            clone(self.regr), X, y, train_index, test_index)
                                                for train_index, test_index in cv.split(X, y))

        predictions = [pred_block for (pred_block, _) in prediction_blocks]
        test_index = np.concatenate([index for (_, index) in prediction_blocks])

        inv_test_index = np.empty(len(test_index), dtype=int)
        inv_test_index[test_index] = np.arange(len(test_index))

        return np.concatenate(predictions)[inv_test_index]

    def gen_serialize_filepath(self, prefix, suffix):
        return self.path.join(
            self.MODELS_SERIALIZING_BASEPATH,
            '.'.join([prefix, self.model_name, suffix])
        )


def _center_scale_xy(X, y, scale=True):
    """From '_center_scale_xy' in sklearn.cross_decomposition.pls_"""
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = y.mean(axis=0)
    y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(y.shape[1])

    return X, y, x_mean, y_mean, x_std, y_std


def _fit_and_predict(estimator, X, y, train_index, test_index):
    """From '_fit_and_predict' in sklearn.model_selection._validation"""
    X_train, y_train = safe_split(X, y, train_index)
    X_test, _ = safe_split(X, y, test_index)

    X_train, y_train, x_mean, y_mean, x_std, y_std = _center_scale_xy(X_train, y_train)
    X_test = (X_test - x_mean) / x_std

    y_pred = estimator.fit(X_train, y_train).predict(X_test)

    return (y_pred * y_std) + y_mean, test_index

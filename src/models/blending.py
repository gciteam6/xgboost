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


class BlendingRegressor:
    def __init__(self, regr):
        self.regr = regr
        self.x_mean = None
        self.y_mean = None
        self.x_std = None
        self.y_std = None

    def set_regressor(self, regr):
        self.regr = regr

    def set_means_stds(self, x_mean, y_mean, x_std, y_std):
        self.x_mean, self.y_mean, self.x_std, self.y_std = \
            x_mean, y_mean, x_std, y_std

    def get_means_stds(self):
        return self.x_mean, self.y_mean, self.x_std, self.y_std


class MyBlender(MyEstimatorBase, PickleMixin):
    def __init__(self, blending_regressor: BaseEstimator, model_name: str, params: dict):
        super().__init__(model_name, params)
        self.blend_model = BlendingRegressor(
            blending_regressor.set_params(**params)
        )
        self.MODELS_SERIALIZING_BASEPATH = self.path.join(self.MODELS_SERIALIZING_BASEPATH,
                                                          MACHINE_LEARNING_TECHNIQUE_NAME)
        self.SERIALIZE_FILENAME_PREFIX = SERIALIZE_FILENAME_PREFIX

    def get_model_instance(self):
        return self.blend_model.regr

    def fit(self, X, y):
        X, y, x_mean, y_mean, x_std, y_std = _center_scale_xy(X, y)
        self.blend_model.set_means_stds(x_mean, y_mean, x_std, y_std)

        fit_model = self.blend_model.regr.fit(X, y.reshape(-1))
        # TODO: locationに関する情報がない
        self.to_pkl(self.blend_model, self.gen_serialize_filepath(self.SERIALIZE_FILENAME_PREFIX, PICKLE_EXTENSION))

        return fit_model

    def predict(self, X):
        blend_model = self.read_pkl(self.gen_serialize_filepath(self.SERIALIZE_FILENAME_PREFIX, PICKLE_EXTENSION))

        x_mean, y_mean, x_std, y_std = blend_model.get_means_stds()
        X = (X - x_mean) / x_std
        y_pred = blend_model.regr.predict(X)

        return (y_pred * y_std) + y_mean

    def cross_val_predict(self, X, y, cv=KFold(n_splits=NUM_FOLDING)):
        """From 'cross_val_predict' in sklearn.model_selection._validation"""
        prediction_blocks = Parallel(n_jobs=-1)(delayed(_fit_and_predict)(
            clone(self.blend_model.regr), X, y, train_index, test_index)
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

# Third-party modules
from xgboost import XGBRegressor
# Hand-made modules
from .base import BloscpackMixin, MyEstimatorBase, PickleMixin

MACHINE_LEARNING_TECHNIQUE_NAME = "xgb"
PICKLE_EXTENSION = "pkl"
SERIALIZE_FILENAME_PREFIX = "fit_model"


class MyXGBRegressor(MyEstimatorBase, BloscpackMixin, PickleMixin):
    def __init__(self, model_name: str, params: dict):
        super().__init__(model_name, params)
        self.regr = self.get_model_instance()
        self.MODELS_SERIALIZING_BASEPATH = self.path.join(self.MODELS_SERIALIZING_BASEPATH,
                                                          MACHINE_LEARNING_TECHNIQUE_NAME)
        self.SERIALIZE_FILENAME_PREFIX = SERIALIZE_FILENAME_PREFIX

    def get_model_instance(self):
        return XGBRegressor(**self.params)

    def fit(self, X, y):
        fit_model = self.regr.fit(X, y, eval_metric="mae")

        pkl_path = self.gen_abspath(
            self.path.join(
                self.MODELS_SERIALIZING_BASEPATH,
                self.gen_serialize_filepath(self.SERIALIZE_FILENAME_PREFIX, PICKLE_EXTENSION)
            )
        )
        self.to_pkl(fit_model, pkl_path)

        return fit_model

    def predict(self, X):
        fit_model = self.read_pkl(
            self.path.join(
                self.MODELS_SERIALIZING_BASEPATH,
                self.gen_serialize_filepath(self.SERIALIZE_FILENAME_PREFIX, PICKLE_EXTENSION)
            )
        )

        return fit_model.predict(X)


if __name__ == '__main__':
    print("My eXtra Gradient Boosting trees !")

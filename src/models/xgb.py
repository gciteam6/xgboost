# Third-party modules
from xgboost import XGBRegressor
# Hand-made modules
from .base import BloscpackMixin, MyEstimatorBase, PickleMixin

MACHINE_LEARNING_TECHNIQUE_NAME = "xgb"
PICKLE_EXTENSION = "pkl"
SERIALIZE_FILEPATH_PREFIX = "fit_model"


class MyXGBRegressor(MyEstimatorBase, BloscpackMixin, PickleMixin):
    def __init__(self, model_name: str, params: dict):
        super().__init__(model_name, params)
        self.MACHINE_LEARNING_TECHNIQUE_NAME = MACHINE_LEARNING_TECHNIQUE_NAME
        self.SERIALIZE_FILEPATH_PREFIX = SERIALIZE_FILEPATH_PREFIX
        self.regr = self.get_model_instance()

    def get_model_instance(self):
        return XGBRegressor(**self.params)

    def fit(self, X, y):
        fit_model = self.regr.fit(X, y, eval_metric="mae")
        self.to_pkl(
            fit_model,
            self.gen_abspath(
                self.path.join(
                    self.MODELS_SERIALIZING_BASEPATH,
                    self.MACHINE_LEARNING_TECHNIQUE_NAME,
                    self.gen_serialize_filepath(
                        self.SERIALIZE_FILEPATH_PREFIX,
                        PICKLE_EXTENSION
                    )
                )
            )
        )

        return fit_model

    def predict(self, X):
        fit_model = self.read_pkl(
            self.path.join(
                self.MODELS_SERIALIZING_BASEPATH,
                MACHINE_LEARNING_TECHNIQUE_NAME,
                self.gen_serialize_filepath(
                    "fit_model",
                    PICKLE_EXTENSION
                )
            )
        )

        return fit_model.predict(X)


if __name__ == '__main__':
    print("My eXtra Gradient Boosting trees !")



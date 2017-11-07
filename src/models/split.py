# Third-party modules
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# Hand-made modules
from .base import PathHandlerBase, BloscpackMixin

KWARGS_READ_CSV = {
    "sep": "\t",
    "header": 0,
    "parse_dates": [0],
    "index_col": 0
}
KWARGS_TO_CSV = {
    "sep": "\t"
}
OBJECTIVE_LABEL_NAMES = ["kwh", ]
Y_TRUE_FILEPATH_PREFIX = "train_y"
Y_TRUE_FILEPATH_SUFFIX = "tsv"


class ValidationSplitHandler(BloscpackMixin):
    def __init__(self):
        super().__init__()

    def separate_and_serialize_validation_index(self,
                                                train_filepath_prefix,
                                                location,
                                                n_splits):
        df = pd.read_csv('.'.join([train_filepath_prefix,
                                   "{l}.tsv".format(l=location)]),
                         **KWARGS_READ_CSV)
        train_index = df.index

        for n_iter, (_, test_index) in enumerate(KFold(n_splits=n_splits).split(train_index)):
            serialized_filepath = '.'.join([train_filepath_prefix,
                                            "index.crossval{i}".format(i=n_iter),
                                            location + ".blp"])
            self.to_blp(
                np.asarray(train_index[test_index]), serialized_filepath
            )


class DatasetSplitHandler(PathHandlerBase):
    def __init__(self):
        super().__init__()
        self.Y_TRUE_FILEPATH_PREFIX = self.path.join(self.PROCESSED_DATA_BASEPATH,
                                                     Y_TRUE_FILEPATH_PREFIX)
        self.Y_TRUE_FILEPATH_SUFFIX = Y_TRUE_FILEPATH_SUFFIX

    def separate_and_save_train_y(self,
                                  train_filepath_prefix,
                                  location):
        df_train = pd.read_csv('.'.join([train_filepath_prefix,
                                         "{l}.tsv".format(l=location)]),
                               **KWARGS_READ_CSV)
        df_train[OBJECTIVE_LABEL_NAMES].to_csv(
            '.'.join([self.Y_TRUE_FILEPATH_PREFIX,
                      location,
                      self.Y_TRUE_FILEPATH_SUFFIX]),
            **KWARGS_TO_CSV
        )


if __name__ == '__main__':
    print("Splitter !")

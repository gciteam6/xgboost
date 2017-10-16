# Third-party modules
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# Hand-made modules
from .base import BloscpackMixin


class ValidationSplitHandler(BloscpackMixin):
    def __init__(self):
        super().__init__()

    def separate_and_serialize_validation_index(self,
                                                train_filepath_prefix,
                                                train_filepath_suffix,
                                                n_splits):
        train_index_filepath = '.'.join([train_filepath_prefix,
                                         "index",
                                         train_filepath_suffix])
        train_index = pd.DatetimeIndex(self.read_blp(train_index_filepath))

        kf = KFold(n_splits=n_splits)

        for n_iter, (_, test_index) in enumerate(kf.split(train_index)):
            serialized_filepath = '.'.join([train_filepath_prefix,
                                            "crossval{i}".format(i=n_iter),
                                            train_filepath_suffix])
            self.to_blp(
                np.asarray(train_index[test_index]), serialized_filepath
            )


if __name__ == '__main__':
    print("Validation index splitter !")

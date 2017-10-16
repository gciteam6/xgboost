# Built-in modules
from glob import glob
# Third-party modules
import pandas as pd
# Hand-made modules
from .base import LocationHandlerBase

AMD_REGEX_DIRNAME = "amd[1-5]"
AMD_COLUMN_NAME_ANNOTATION = "amd"
KWARGS_READ_CSV_AMD_MASTER = {
    "index_col": 0,
}
KWARGS_READ_CSV_AMD_LOG = {
    "index_col": 0,
}


class AmedasHandler(LocationHandlerBase):
    def __init__(self,
                 amd_master_filepath,
                 amd_file_prefix="amd_",
                 amd_file_suffix=".tsv"):
        super().__init__(amd_master_filepath, **KWARGS_READ_CSV_AMD_MASTER)
        self.amd_file_prefix = amd_file_prefix
        self.amd_file_suffix = amd_file_suffix
        self.AMD_REGEX_DIRNAME = AMD_REGEX_DIRNAME
        self.AMD_COLUMN_NAME_ANNOTATION = AMD_COLUMN_NAME_ANNOTATION

    def read_tsv(self, path_or_buf):
        df_ret = pd.read_csv(path_or_buf, **self.gen_read_csv_kwargs(KWARGS_READ_CSV_AMD_LOG))
        df_ret.index = self.parse_datetime(pd.Series(df_ret.index).apply(str))

        return df_ret

    def to_tsv(self, df, path_or_buf, **kwargs):
        df.to_csv(path_or_buf, **self.gen_to_csv_kwargs(kwargs))

    def gen_filepath_list(self, aid_list):
        amd_regex_filepath_list = [
            self.path.join(
                self.INTERIM_DATA_BASEPATH,
                self.AMD_REGEX_DIRNAME,
                self.amd_file_prefix + str(aid) + self.amd_file_suffix
            ) for aid in aid_list
        ]

        return [
            amd_file \
            for amd_regex_filepath in amd_regex_filepath_list \
            for amd_file in glob(amd_regex_filepath)
        ]

    def retrieve_data(self, filepath_list, location_list):
        if len(filepath_list) < 1:
            raise ValueError("Empty list ?")

        df_ret = self.read_tsv(filepath_list[0])
        df_ret.columns = self.add_annotations_to_column_names(
            df_ret, self.AMD_COLUMN_NAME_ANNOTATION, location_list[0]
        )

        if len(filepath_list) > 1:
            for filepath, location in zip(filepath_list[1:], location_list[1:]):
                df_temp = self.read_tsv(filepath_list[0])
                df_temp.columns = self.add_annotations_to_column_names(
                    df_temp, self.AMD_COLUMN_NAME_ANNOTATION, location
                )

                df_ret = df_ret.merge(
                    df_temp, **self.KWARGS_OUTER_MERGE
                )

        return df_ret


if __name__ == '__main__':
    print("Amedas!")

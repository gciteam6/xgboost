# Built-in modules
from glob import glob
from zipfile import ZipFile
# Hand-made modules
from .base import PathHandlerBase


class Unzipper(PathHandlerBase):
    def __init__(self):
        super().__init__()

    def unzip(self, zip_filepath, extract_dirpath):
        extract_dirpath = self.gen_abspath(extract_dirpath)

        file_handler = ZipFile(zip_filepath, 'r')
        file_handler.extractall(extract_dirpath)

        file_handler.close()

    def gen_unzip_filepath_list(self, unzip_dirpath):
        zipfile_regex_filepath = self.path.join(self.path.abspath(unzip_dirpath), "*.zip")
        return glob(zipfile_regex_filepath)


if __name__ == '__main__':
    print("unzip!")

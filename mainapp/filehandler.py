import pandas as pd
import os


class dataset:

    def __init__(self, data):

        self.data = data

    def read_data(self):
        pass


def read_file(path):
    type_parsers = {
        '.xls': pd.read_excel,
        '.xlsx': pd.read_excel,
        '.csv': pd.read_csv,
    }

    file_type = os.path.splitext(path)[1]

    if file_type in type_parsers:
        dt = dataset(
            type_parsers[file_type]('storage/{}'.format(path), encoding='latin-1')
        )
        return dt
    else:
        return 'BB'





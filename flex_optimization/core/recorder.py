import copy
from abc import ABC, abstractmethod

import pandas as pd


class Recorder(ABC):
    SETUP = 0
    RUNNING = 1
    EVALUATION = 2
    STOP = 3
    FINISH = 4
    NOTES = 5
    WARNING = 6

    def __init__(self, problem=None, method=None):
        self.problem = problem
        self.method = method
        self.data = []
        self._df = None
        self.best_result = None
        self._error = None

    def _error_exit(self, e):
        """ Here to do something if error occurs during optimization. """
        raise e

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            if len(self.data) == 0:
                raise Exception("Error occurred. No data to generate data frame.")
            columns = self.problem.variable_names
            columns += [f"inter_{i}" for i in range(len(self.data[0]) - self.problem.num_variables - 1)]
            columns += ["metric"]
            self._df = pd.DataFrame(self.data, columns=columns)

        return self._df

    @abstractmethod
    def record(self, type_: int, *args, **kwargs):
        ...

    def save(self, file_name: str):
        self.df.to_csv(f"{file_name}.csv")
        self._save_self(file_name)

    def _save_self(self, file_name):
        import pickle
        obj = copy.copy(self)
        del obj.data
        del obj._df
        with open(f"{file_name}.pickle", "wb") as file:
            pickle.dump(obj, file)

    @classmethod
    def load(cls, file_name: str):
        import pickle
        with open(f"{file_name}.pickle", "rb") as file:
            obj = pickle.load(file)

        csv_ = pd.read_csv(f"{file_name}.csv", index_col=0)
        obj._df = csv_

        if hasattr(obj, "_error") and obj._error is not None:
            import warnings
            from flex_optimization.core.utils import custom_formatwarning
            warnings.formatwarning = custom_formatwarning
            warnings.warn(
                "\033[31m" + "The data you loaded was from an optimization that had an error. See 'self._error'"
                "for details." + "\033[0m")
        return obj

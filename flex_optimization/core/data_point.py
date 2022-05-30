from typing import Any

import numpy as np


def shape_check(obj):
    if isinstance(obj, (float, int, str)):
        obj = [obj]
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    out = []
    for item in obj:
        if isinstance(item, (list, tuple, np.ndarray)):
            if len(item) == 1:
                out.append(item[0])
                continue
            else:
                raise ValueError(f"Invalid object in array: {obj}")
        out.append(item)

    return out


def result_metric_same(result, metric) -> bool:
    try:
        if metric == result:
            return True
    except ValueError:
        pass
    return False


class DataPoint:
    def __init__(self,
                 point: float | int | str | list | tuple | np.ndarray,
                 result: float | int | str | list | tuple | np.ndarray,
                 metric: float | int | str | list | tuple | np.ndarray,
                 iteration: int = None
                 ):
        """

        Parameters
        ----------
        point: float|int|str|list|tuple|np.ndarray
            must be a single dimension
        result: float|int|str|list|tuple|np.ndarray
            must be a single dimension
        metric: float|int|str|list|tuple|np.ndarray
            must be a single dimension
        iteration: int
            iteration in algorithm

        """
        self._point = None
        self.point = point

        self._has_metric = not result_metric_same(result, metric)

        self._result = None
        self.result = result

        self._metric = None
        self.metric = metric

        self.iteration = iteration

    def __repr__(self) -> str:
        mes = f" {self.point} --> {self.result}"
        if self.has_metric:
            mes += f" --> {self.metric}"
        return mes

    @property
    def point(self) -> list[float | int | str]:
        return self._point

    @point.setter
    def point(self, point):
        self._point = shape_check(point)

    @property
    def result(self) -> list[Any]:
        return self._result

    @result.setter
    def result(self, result):
        self._result = shape_check(result)

    @property
    def metric(self) -> list[float | int]:
        return self._metric

    @metric.setter
    def metric(self, metric):
        self._metric = shape_check(metric)

    @property
    def has_metric(self) -> bool:
        """ returns True if result and metric are the same. """
        return self._has_metric

    @property
    def data_chunk(self) -> list[(int, float, str)]:
        out = []
        if self.iteration is not None:
            out.append(self.iteration)
        out += self.point
        out += self.result
        if self.has_metric:
            out += self.metric
        return out

import copy
import multiprocessing
from functools import wraps

import numpy as np

from flex_optimization.core.recorder import Recorder


class PoolHandler:
    def __init__(self, func: callable, pool_size: int, pool_points: list | tuple | np.ndarray,
                 callback: callable, args: tuple = ()):
        self.func = func
        self.pool_size = pool_size
        if isinstance(pool_points, np.ndarray):
            pool_points = pool_points.tolist()
        self.pool_points = copy.deepcopy(pool_points)
        self.callback = callback
        self.args = args
        self._results: list[multiprocessing.pool.AsyncResult] = []
        self._process_running = 0

    def run(self):
        with multiprocessing.Pool(self.pool_size) as pool:
            while True:
                # start new process
                if self._process_running < self.pool_size and len(self.pool_points) != 0:
                    self._start_new_process(pool)

                # check for completed process
                self._result_done_check()

                # Exit statements
                if len(self.pool_points) == 0 and len(self._results) == 0:
                    break

    def _start_new_process(self, pool):
        point = self.pool_points.pop()
        result = pool.apply_async(self.func, kwds=dict(point=point))
        self._results.append(result)
        self._process_running += 1

    def _result_done_check(self):
        for i, result in enumerate(self._results):
            if result.ready():
                break
        else:
            return

        result = self._results.pop(i)
        self._result_done(result)

    def _result_done(self, result):
        self._process_error_check(result)
        data = result.get()
        self.callback(data)
        self._process_running -= 1

    @staticmethod
    def _process_error_check(result):
        if result.successful():
            return  # process exited normally
        result.get()
        raise multiprocessing.ProcessError(f"Process exited unexpectedly with error code.")


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


def save_if_error(func):
    """
    This function decorator will trigger the recorder to save the data if there is an error.
    """
    @wraps(func)
    def _save_if_error(*args, **kwargs):  # first arg (it is self)
        recorder: Recorder = args[0].recorder
        try:
            return func(*args, **kwargs)

        except KeyboardInterrupt as e:  # keyboard interrupt is not an Exception
            recorder._error_exit(e)

        except Exception as e:
            print(e)
            recorder._error_exit(e)

    return _save_if_error

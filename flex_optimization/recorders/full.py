import time
from datetime import datetime

from flex_optimization.core.data_point import DataPoint
from flex_optimization.core.recorder import Recorder
from flex_optimization.core.logger_ import logger


class RecorderFull(Recorder):
    def __init__(self, problem=None, method=None):
        self.start_time = None
        self.end_time = None
        self._first_eval = True
        self.num_data_points: int = 0
        super().__init__(problem, method)
        logger.setLevel(logger.DEBUG)

    @property
    def duration(self):
        return self.end_time - self.start_time

    def _error_exit(self, e):
        """ Save data if something goes wrong in the middle of the optimization. """
        from datetime import datetime

        self._error = e
        if len(self.data) > 1:
            filename = f"data_error_export_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            logger.error(f"Error Occurred!!!!!!!! Data saved as '{filename}'")
            self.save(filename)
        else:
            logger.error(f"Error Occurred!!!!!!!!")

        raise e

    def record(self, type_: int, **kwargs):
        if type_ == self.EVALUATION:
            self._record_evaluation(**kwargs)
        elif type_ == self.SETUP:
            self._record_setup()
        elif type_ == self.RUNNING:
            self._record_running()
        elif type_ == self.STOP:
            self._record_stop(**kwargs)
        elif type_ == self.FINISH:
            self._record_finish()
        elif type_ == self.NOTES:
            self._record_notes(**kwargs)
        elif type_ == self.WARNING:
            self._record_warning(**kwargs)
        elif type_ == self.ERROR:
            self._record_error(**kwargs)
        else:
            raise NotImplementedError("The full recorder should implement everything!")

    def _record_setup(self):
        # log problem
        logger.info(f"{type(self.problem).__name__} | {repr(self.problem)}")

        # log variables
        for var in self.problem.variables:
            logger.info(f"{type(var).__name__} | {repr(var)}")

        # log method
        logger.info(f"{type(self.method).__name__} | {repr(self.method)}")

    def _record_running(self):
        self.start_time = datetime.now()
        logger.info(f"\nOptimization Running | start:{self.start_time}")

    def _record_evaluation(self, data_point: DataPoint):
        show_iteration = True if data_point.iteration is not None else False

        if self._first_eval:
            self._first_eval = False
            self._create_header(data_point.has_metric, show_iteration)

        # save data
        self.data.append(data_point)

        # send data to logger
        mes = f"{self.num_data_points} |"
        if show_iteration:
            mes += f" {self.method.iteration_count} |"
        mes += data_point.__repr__()
        logger.monitor(mes)
        self.num_data_points += 1
        self._up_to_date = False

    def _create_header(self, show_metric: bool = False, show_iteration: bool = False):
        mes = "counter |"
        if show_iteration:
            mes += " iteration | "
        mes += f"{self.problem.variable_names} --> result"
        if show_metric:
            mes += " --> metric"
        logger.monitor(mes)
        logger.monitor("-" * len(mes))

    @staticmethod
    def _record_stop(text: str):
        logger.info(text)

    def _record_finish(self):
        self.end_time = datetime.now()
        time.sleep(0.1)  # to ensure all evaluations print first

        logger.info(f"Calculation time: {self.duration}")
        logger.info(f"\nBest Result: {self.best_result}")

    @staticmethod
    def _record_notes(text: str):
        logger.info(text)

    @staticmethod
    def _record_warning(text: str):
        logger.warn(text)

    @staticmethod
    def _record_error(text: str):
        logger.error(text)

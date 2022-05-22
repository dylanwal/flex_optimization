
from flex_optimization.problem_statement import StopCriteria


class StopFunctionEvaluation(StopCriteria):
    def __init__(self, num_eval: int = 10):
        self.num_eval = num_eval
        self.current_eval = 0

    def __repr__(self):
        return f"{type(self).__name__} | num_eval: {self.current_eval}"

    def evaluate(self, *args, **kwargs) -> bool:
        self.current_eval += 1
        if self.current_eval == self.num_eval:
            return False

        return True

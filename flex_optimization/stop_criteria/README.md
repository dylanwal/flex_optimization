# Stopping Criteria

----
----

This directory contains collection of stopping criteria for optimization.

----

# StopFunctionEvaluation


    Stop Criteria: Function Evaluation

    Stop the algorithm after 'num_eval' function evaluations.

    


----

# StopIterationEvaluation


    Stop Criteria: Function Iteration

    Stop the algorithm after 'num_eval' iterations.
    * an iteration may be 1 function evaluation; but it may not be.

    


----

# StopComputationTime


    Stop Criteria: Computation Time

    Stop the algorithm after 'duration' (time that has passed).
    * It will finish current iteration before stopping.

    


----

# StopRelativeChange


    Stop Criteria: Relative Change

    Stop the algorithm when the best value (relative to the prior best value) has not changed more than the
        'cut_off_value' for 'cut_off_steps'.
    * setting the 'cut_off_steps' to 1 will cause the algorithm to stop at the first instance when the best value
        doesn't increase by the 'cut_off_value
    * setting the 'cut_off_steps' to >1 will allow the algorithm to continue for 'cut_off_steps' without any
        improvement before stopping. The counter resets each time a new best value bigger than the
        'cut_off_value' is found.

    


----

# StopAbsoluteChange


    Stop Criteria: Absolute Change

    Stop the algorithm when the best value has not changed more than the 'cut_off_value' for 'cut_off_steps'.
    * setting the 'cut_off_steps' to 1 will cause the algorithm to stop at the first instance when the best value
        doesn't increase by the 'cut_off_value
    * setting the 'cut_off_steps' to >1 will allow the algorithm to continue for 'cut_off_steps' without any
        improvement before stopping. The counter resets each time a new best value bigger than the
        'cut_off_value' is found.

    


----

# StopRate


    Stop Criteria: Rate

    Stop the algorithm after 'cut_off_steps' iterations of the slope of the past 'prior_steps' falls below the
    'cut_off_rate'.

    


----


import time

import flex_optimization as fo
import flex_optimization.methods as fo_m
import flex_optimization.stop_criteria as fo_s
import flex_optimization.problems as fo_p

fo.logger.setLevel(fo.logger.MONITOR)


def main():
    problem = fo.Problem(
        func=fo_p.two_d_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(x_o=0.2342, y_o=0.1234, sigma_x=3),
        optimization_type=True
    )

    # STOP CRITERIA
    method = fo_m.MethodRandom(problem, fo_s.StopFunctionEvaluation(100))
    for i in range(10):
        method.run_steps(step=1)
        # do something between each step
        time.sleep(0.5)
        print(f"step{i} done")

    vis = fo.VizOptimization(problem, method.data)
    fig = vis.plot_3d_vis()
    fig.show()


if __name__ == "__main__":
    main()

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

    # method = fo_m.MethodFactorial(problem=problem, levels=10)
    # method = fo_m.MethodRandom(problem, fo_s.StopFunctionEvaluation(100))
    # method = fo_m.MethodRandom(problem, fo_s.StopComputationTime(1))
    # method = fo_m.MethodRandom(problem, [fo_s.StopComputationTime(1), fo_s.StopFunctionEvaluation(100)])
    # method = fo_m.MethodRandom(problem, [[fo_s.StopComputationTime(0.1), fo_s.StopFunctionEvaluation(1000)]])
    # method = fo_m.MethodRandom(problem, fo_s.StopRelativeChange(cut_off_steps=10))
    # method = fo_m.MethodRandom(problem, fo_s.StopAbsoluteChange(cut_off_value=0.05, cut_off_steps=10))
    method = fo_m.MethodRandom(problem, fo_s.StopRate(cut_off_rate=0.05, prior_steps=5, cut_off_steps=2))
    method.run()

    vis = fo.OptimizationVis(problem, method.data)
    fig = vis.plot_3d_vis()
    fig.show()


if __name__ == "__main__":
    main()

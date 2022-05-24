import flex_optimization as fo
import flex_optimization.methods as fo_m
import flex_optimization.stop_criteria as fo_s
import flex_optimization.problems as fo_p

fo.logger.setLevel(fo.logger.DEBUG)


def main():
    problem = fo.Problem(
        func=fo_p.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.2342, 0.1234], sigma=[1, 3]),
        optimization_type=True
    )

    # STOP CRITERIA
    # method = fo_m.MethodRandom(problem, fo_s.StopComputationTime(1))
    # method = fo_m.MethodRandom(problem, [fo_s.StopComputationTime(1), fo_s.StopFunctionEvaluation(100)])
    # method = fo_m.MethodRandom(problem, [[fo_s.StopComputationTime(0.1), fo_s.StopFunctionEvaluation(1000)]])
    # method = fo_m.MethodRandom(problem, fo_s.StopRelativeChange(cut_off_steps=10))
    # method = fo_m.MethodRandom(problem, fo_s.StopAbsoluteChange(cut_off_value=0.05, cut_off_steps=10))
    # method = fo_m.MethodRandom(problem, fo_s.StopRate(cut_off_rate=0.05, prior_steps=5, cut_off_steps=2))

    # METHODS
    # method = fo_m.MethodFactorial(problem=problem, levels=10)
    # method = fo_m.MethodCovary(problem=problem, levels=10)
    # method = fo_m.MethodMultiCovary(problem=problem, levels=10)
    # method = fo_m.MethodStar(problem=problem, levels=3)
    # method = fo_m.MethodRandom(problem, fo_s.StopFunctionEvaluation(100))
    # method = fo_m.MethodSobol(problem, fo_s.StopFunctionEvaluation(100))
    # method = fo_m.MethodLatinHypercube(problem, fo_s.StopFunctionEvaluation(100))
    # method = fo_m.MethodHalton(problem, fo_s.StopFunctionEvaluation(100))
    # method = fo_m.MethodMultiNormal(problem, fo_s.StopFunctionEvaluation(100))
    # method = fo_m.MethodBODragon(problem, stop_criteria=fo_s.StopFunctionEvaluation(20))
    method = fo_m.MethodBFGS(problem, stop_criteria=fo_s.StopFunctionEvaluation(20), x0=[3, 3])

    # multiprocessing
    # method = fo_m.MethodBODragon(problem, stop_criteria=fo_s.StopFunctionEvaluation(20), multiprocess=True,
    #                              options=dict(build_new_model_every=3))

    method.run()

    vis = fo.OptimizationVis(problem, method.data)
    fig = vis.plot_3d_vis()
    fig.show()


if __name__ == "__main__":
    main()

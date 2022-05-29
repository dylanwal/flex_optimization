import flex_optimization as fo


def main():
    problem = fo.Problem(
        func=fo.problems.nd_gaussian,
        variables=[fo.ContinuousVariable(-5, 5, name="x"), fo.ContinuousVariable(-5, 5, name="y")],
        kwargs=dict(center=[0.523423, 0.1324], sigma=[1, 3]),
        type_=fo.OptimizationType.MAX
    )

    # STOP CRITERIA
    # method = fo.methods.MethodRandom(problem, fo.stop_criteria.StopComputationTime(0.05))
    # method = fo.methods.MethodRandom(problem, [fo.stop_criteria.StopComputationTime(0.11), fo.stop_criteria.StopFunctionEvaluation(100)])  # or
    # method = fo.methods.MethodRandom(problem, [[fo.stop_criteria.StopComputationTime(0.1), fo.stop_criteria.StopFunctionEvaluation(1000)]])  # and
    # method = fo.methods.MethodRandom(problem, fo.stop_criteria.StopRelativeChange(cut_off_value=1E-10, cut_off_steps=10))   # no improvement for 10 steps
    # method = fo.methods.MethodRandom(problem, fo.stop_criteria.StopRelativeChange(cut_off_value=0.4, cut_off_steps=1))  # stop imidately when there is no relative imporvement
    # method = fo.methods.MethodRandom(problem, fo.stop_criteria.StopAbsoluteChange(cut_off_value=0.05, cut_off_steps=10))
    # method = fo.methods.MethodRandom(problem, fo.stop_criteria.StopRate(cut_off_rate=0.05, prior_steps=5, cut_off_steps=2))

    # METHODS
    # method = fo.methods.MethodFactorial(problem=problem, levels=10)
    # method = fo.methods.MethodCovariance(problem=problem, levels=10)
    # method = fo.methods.MethodMultiCovariance(problem=problem, levels=10)
    # method = fo.methods.MethodStar(problem=problem, levels=3)
    
    # method = fo.methods.MethodRandom(problem, fo.stop_criteria.StopFunctionEvaluation(100))
    # method = fo.methods.MethodSobol(problem, fo.stop_criteria.StopFunctionEvaluation(100))
    # method = fo.methods.MethodLatinHypercube(problem, fo.stop_criteria.StopFunctionEvaluation(100))
    # method = fo.methods.MethodHalton(problem, fo.stop_criteria.StopFunctionEvaluation(100))
    # method = fo.methods.MethodMultiNormal(problem, fo.stop_criteria.StopFunctionEvaluation(100))
    # method = fo.methods.MethodBODragon(problem, stop_criteria=fo.stop_criteria.StopFunctionEvaluation(10))
    # method = fo.methods.MethodBFGS(problem, stop_criteria=fo.stop_criteria.StopFunctionEvaluation(20), x0=[3, 3])

    # multiprocessing
    # method = fo.methods.MethodFactorial(problem=problem, levels=10, multiprocess=True)
    method = fo.methods.MethodBODragon(problem, stop_criteria=fo.stop_criteria.StopFunctionEvaluation(12), multiprocess=True,
                                 options=dict(build_new_model_every=3))

    method.run()

    vis = fo.OptimizationVis(method.recorder)
    fig = vis.plot_3d_vis()
    fig.show()


if __name__ == "__main__":
    main()

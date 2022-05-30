
import numpy as np

from flex_optimization import OptimizationType
from flex_optimization.problems import ProblemClassification


def sphere(args: list[float]) -> float:
    """
    Sphere function

    References
    ----------


    Features:
    --------
    *  smooth optimization
    * single minima at center
    * typical x range [-5, 5]

    Parameters
    ----------
    args: list[float]
        [x, y, z, ...] (length determines dimensionality)

    Returns
    -------
    return: float
        z value

    """
    d = len(args)

    sum_ = 0
    for i in range(d):
        sum_ += args[i]**2

    return sum_


def goal(d: int):
    return np.zeros(d)


classification_class = ProblemClassification(
    name="sphere",
    func=sphere,
    type_=OptimizationType.MIN,
    global_goal=goal,
    range_=(-5, 5),
    local_min=0,
    num_dim=(1, float('inf')),
    convex=True,
    roughness=0,
    symmetric=True
)


def local_run():
    n = 100
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    xx, yy = np.meshgrid(x, y)

    xx = xx.T.reshape(n*n)
    yy = yy.T.reshape(n*n)
    zz = sphere([xx, yy])

    import plotly.graph_objs as go
    fig = go.Figure(go.Surface(x=x, y=y, z=zz.reshape(n, n).T))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(color="white", size=5)))

    # fig.write_image("imgs//sphere.svg")
    fig.show()


def local_run2():
    n = 20
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    z = np.linspace(-5, 5, n)
    xx, yy, zz = np.meshgrid(x, y, z)

    xx = xx.T.reshape(n*n*n)
    yy = yy.T.reshape(n*n*n)
    zz = zz.T.reshape(n*n*n)
    aa = sphere([xx, yy, zz])

    import plotly.express as px
    import pandas as pd
    df = pd.DataFrame(np.stack((xx, yy, zz, aa), axis=1))
    fig = px.scatter_3d(df, x=0, y=1, z=2, color=3)
    fig.show()


if __name__ == "__main__":
    local_run()

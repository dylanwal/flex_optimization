
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd


def star(d: int, center: bool = True) -> np.ndarray:
    """
    Create the star points of various design matrices

    Parameters
    ----------
    d : int
        The number of dimensions
    center: bool
        Add center point

    Return
    ------
    points: np.ndarray
        points of d-dimensional star

    """
    # Create the actual matrix now.
    points = np.zeros((2 * d, d))
    for i in range(d):
        points[2 * i:2 * i + 2, i] = [-1, 1]

    if center:
        points = np.insert(points, 0, np.zeros(d), axis=0)

    return points


def star_levels(d: int, center: bool = True, levels: int = 2) -> np.ndarray:
    star_points = star(d, center)
    if center:
        center_point = star_points[0]
        star_points = star_points[1:]

    points = None
    space = 1/levels
    for level in range(1, levels+1):
        temp_points = star_points * (space * level)
        if points is None:
            points = temp_points
        else:
            points = np.vstack((points, temp_points))

    if center:
        points = np.insert(points, 0, center_point, axis=0)

    return points


a = star_levels(4, levels=3)
df = pd.DataFrame(a)
df[0] = df[0] + df[3]*0.03
# fig = go.Figure(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2]))
fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2], color=df.columns[3])
fig.show()

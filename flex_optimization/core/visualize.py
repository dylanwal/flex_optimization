import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

from flex_optimization.core.recorder import Recorder


class VizOptimization:
    def __init__(self, recorder: Recorder, true_func: callable = None):
        self.recorder = recorder
        self.df: pd.DataFrame = self.recorder.df
        self.true_func = true_func

    def plot_by_expt(self) -> go.Figure:
        fig = go.Figure()
        for col in self.df.columns:
            fig.add_trace(go.Scatter(x=self.df.index, y=self.df[col], name=col,
                                     mode="markers+lines", line=dict(width=1)
                                     ))

        return fig

    def _add_true_function(self, fig):
        if not len(self.recorder.problem.variables) == 2:
            print("Only 2 variables are supported for true function plotting. ")
            return

        range_ = []
        for var in self.recorder.problem.variables:
            range_.append([var.min_, var.max_])

        n = 30
        x = np.linspace(range_[0][0], range_[0][1], n)
        y = np.linspace(range_[1][0], range_[1][1], n)
        xx, yy = np.meshgrid(x, y)

        xx = xx.T.reshape(n*n)
        yy = yy.T.reshape(n*n)
        zz = self.true_func([xx, yy])
        fig.add_trace(go.Surface(x=x, y=y, z=zz.reshape(n, n).T))

    def plot_3d_vis(self, indept_var: list[str] = None) -> go.Figure:
        if len(self.df) < 3:
            raise ValueError("2 independent variable are required to visualized in 3D.")
        if indept_var is None:
            cols = list(self.recorder.problem.variable_names[:2]) + ["metric"]
        else:
            if len(indept_var) != 2:
                raise ValueError("Two independent variable names must be provided")

            cols = indept_var + ["metric"]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=self.df[cols[0]], y=self.df[cols[1]], z=self.df[cols[2]],
                                   mode="markers+lines", line=dict(width=1)
                                   ))

        if self.true_func is not None:
            self._add_true_function(fig)

        return fig

    def plot_4d_vis(self, indept_var: list[str] = None, metric_vis: str = "color") -> go.Figure:
        if len(self.df) < 4:
            raise ValueError("3 independent variable are required to visualized in 4D.")
        if indept_var is None:
            cols = list(self.recorder.problem.variable_names[:3])
        else:
            if len(indept_var) != 3:
                raise ValueError("Three independent variable names must be provided")
            cols = indept_var

        if metric_vis == "color":
            cols.append("metric")
        elif metric_vis == "z":
            cols.insert(2, "metric")
        else:
            raise ValueError("Invalid metric_vis value.")

        fig = px.scatter_3d(self.df, x=cols[0], y=cols[1], z=cols[2], color=cols[3])
        return fig

    def plot_5d_vis(self, indept_var: list[str] = None, metric_vis: str = "color") -> go.Figure:
        if len(self.df) < 5:
            raise ValueError("4 independent variable are required to visualized in 5D.")
        if indept_var is None:
            cols = list(self.recorder.problem.variable_names[:3])
        else:
            if len(indept_var) != 3:
                raise ValueError("Three independent variable names must be provided")
            cols = indept_var

        if metric_vis == "color":
            cols.insert(3, "metric")
        elif metric_vis == "z":
            cols.insert(2, "metric")
        elif metric_vis == "size":
            cols.append("metric")
        else:
            raise ValueError("Invalid metric_vis value.")

        fig = px.scatter_3d(self.df, x=cols[0], y=cols[1], z=cols[2], color=cols[3], size=cols[4])

        return fig

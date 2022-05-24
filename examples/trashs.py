
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

class te:
    def __init__(self):
        self.flag =False

    def fun(self):
        a = [0,1,2,3,4,5]
        for i in a:
            yield i
        self.flag = True

T = te()

for i in te.fun(te):
    print(i)

print(T.flag)
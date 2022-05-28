import time

import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd


def fun(a=1):

    for i in range(20000):
        a = a + a


if __name__ == "__main__":
    try:
        for i in range(100):
            fun(1)
            time.sleep(0.3)
    except KeyboardInterrupt as e:
        exit(e)
        print("keysssss")

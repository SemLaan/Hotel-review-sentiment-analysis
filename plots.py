import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


dist = np.random.multinomial(5, [0.5, 0.5], size=20)

probs = [[i, i] for i in range(len(dist))]

for row in range(len(dist)):
    probs[row][0] = dist[row, 0] / dist[row].sum()
    probs[row][1] = dist[row, 1] / dist[row].sum()

go.Figure(
    data=go.Heatmap(
        z=list(np.array(probs).transpose()),
        x=['word ' + str(i+1) for i in range(len(probs))],
        y=['positive', 'negative'],
        colorscale=[[0.0, "rgb(255,255,255)"], [1.0, "rgb(49,54,149)"]],
        colorbar=dict(
            title="probability"
        )
    )
)



from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, dcc, html
from inverse_folding.data.pdb_to_graph import pdb_to_graph
from inverse_folding.utils.visualization import plot_graph

HOST = "127.0.0.1"
PORT = "8050"
DEFAULT_RADIUS = 0.0
DEFAULT_NEIGHBORS = 1
app = Dash(__name__)
pdb_file = Path("data/pdb/1a6t.pdb")
graph = pdb_to_graph(pdb_file, ["H", "L"], radius=DEFAULT_RADIUS, max_neighbors=DEFAULT_NEIGHBORS)
fig = plot_graph(graph, DEFAULT_RADIUS, DEFAULT_NEIGHBORS)
# fig.layout
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    ),
    showlegend=False,
    width=1200,
    height=800,
)

app.layout = html.Div(
    [
        html.H1("Построение графа по 3D-координатам"),
        html.Div(
            [
                html.P("Максимальное расстояние между атомами, Å"),
                dcc.Slider(
                    id="radius",
                    min=0,
                    max=20,
                    marks={i: str(i) for i in range(0, 21, 4)},
                    value=DEFAULT_RADIUS,
                ),
                html.P("Максимальное количество соседей"),
                dcc.Slider(
                    id="neighbors",
                    min=1,
                    max=30,
                    step=1,
                    marks={i: str(i) for i in range(0, 31, 5)},
                    value=DEFAULT_NEIGHBORS,
                ),
            ],
            style={"display": "inline-block", "width": "22%", "verticalAlign": "top"},
        ),
        html.Div(
            [
                dcc.Graph(id="graph", figure=fig),
            ],
            style={"display": "inline-block", "width": "78%", "verticalAlign": "top"},
        ),
    ]
)


@app.callback(
    Output("graph", "figure"),
    Input("radius", "value"),
    Input("neighbors", "value"),
)
def render_graph(radius: float, neighbors: int) -> go.Figure:
    patched_figure = Patch()
    new_fig = plot_graph(graph, radius, neighbors)
    patched_figure["data"] = new_fig.data
    return patched_figure


app.run_server(debug=True)

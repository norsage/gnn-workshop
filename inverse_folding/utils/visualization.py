import itertools

import plotly.graph_objects as go
from inverse_folding.utils.constants import INDEX_TO_LETTER
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph


def plot_graph(graph: Data, radius: float = 10.0, max_neighbors: int = 5) -> go.Figure:
    assert graph.pos is not None
    graph.edge_index = radius_graph(
        x=graph.pos,
        r=radius,
        max_num_neighbors=max_neighbors,
    )
    row, col = graph.edge_index
    graph.edge_attr = (graph.pos[row] - graph.pos[col]).norm(dim=-1)
    unique_edges, unique_weights = graph.edge_index, graph.edge_attr
    # unique_edges, unique_weights = coalesce(
    #     graph.edge_index.sort(0)[0],
    #     graph.edge_attr,
    #     sort_by_row=False,
    # )
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=graph.pos[:, 0],
                y=graph.pos[:, 1],
                z=graph.pos[:, 2],
                mode="markers+text",
                hoverinfo="text",
                text=[INDEX_TO_LETTER[y.item()] for y in graph.y],
                marker=dict(
                    size=5,
                    color=graph.y,
                    colorscale="Viridis",
                    opacity=0.8,
                ),
            ),
            go.Scatter3d(
                x=list(
                    itertools.chain(
                        *((graph.pos[i, 0], graph.pos[j, 0], None) for i, j in unique_edges.T)
                    )
                ),
                y=list(
                    itertools.chain(
                        *((graph.pos[i, 1], graph.pos[j, 1], None) for i, j in unique_edges.T)
                    )
                ),
                z=list(
                    itertools.chain(
                        *((graph.pos[i, 2], graph.pos[j, 2], None) for i, j in unique_edges.T)
                    )
                ),
                mode="lines",
                # text=unique_weights,  # error
                text=list(
                    itertools.chain(
                        *((f"{d:.3f}Å", f"{d:.3f}Å", None) for d in unique_weights.tolist())
                    )
                ),
                hoverinfo="text",
            ),
        ]
    )
    return fig

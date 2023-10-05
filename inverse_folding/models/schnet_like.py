from typing import Sequence

from inverse_folding.models.lightning import BaseModel
from torch import Tensor, nn
from torch_geometric.nn.models.schnet import (
    GaussianSmearing,
    InteractionBlock,
    RadiusInteractionGraph,
)
from inverse_folding.data.transforms import ResidueNumberingEncoding, DummyEncoding


class NodeClassification(BaseModel):
    def __init__(
        self,
        embedding_dim: int = 64,
        # max_length: int = 512,  # depends on ResidueNumberingEncoding used, set to current maximum
        cutoff: float = 10.0,
        max_neighbors: int = 15,
        num_gaussians: int = 32,
        num_filters: int = 8,
        num_layers: int = 3,
        output_dim: int = 20,
        transform: ResidueNumberingEncoding = DummyEncoding(),
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=transform.embedding_size, embedding_dim=embedding_dim)
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_neighbors)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions: Sequence[InteractionBlock] = nn.ModuleList(  # type: ignore[assignment]
            [
                InteractionBlock(embedding_dim, num_gaussians, num_filters, cutoff)
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        residue_number = x
        h = self.embedding(residue_number)
        edge_index, edge_weight = self.interaction_graph.forward(pos, batch)
        edge_attr = self.distance_expansion.forward(edge_weight)

        for interaction in self.interactions:
            h = h + interaction.forward(h, edge_index, edge_weight, edge_attr)

        logits = self.fc(h)

        return logits

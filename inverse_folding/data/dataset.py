from pathlib import Path

import pandas as pd
from inverse_folding.data.pdb_to_graph import pdb_to_graph
from inverse_folding.data.transforms import ResidueNumberingEncoding
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm


class NodeClassificationDataset(Dataset):
    data: list[Data]

    def __init__(
        self,
        subset_csv: Path,
        datadir: Path,
        resnum_encoder: ResidueNumberingEncoding,
        radius: float = 10.0,
        max_neighbors: int = 15,
    ) -> None:
        # read subset CSV file
        df = pd.read_csv(subset_csv)

        # read and tranform all complexes to graphs
        self.data = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            pdb_path = datadir / f"{row['pdb']}.pdb"
            # antibody_chains = [
            #     chain_id.strip() for chain_id in str(row["antibody_chains"]).split(" ")
            # ]
            complex_graph = pdb_to_graph(
                pdb=pdb_path,
                antibody_chains=["H", "L"],
                radius=radius,
                max_neighbors=max_neighbors,
                resnum_encoder=resnum_encoder,
            )

            self.data.append(complex_graph)

    def __getitem__(self, index: int) -> Data:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

from pathlib import Path
from typing import cast

import torch
from Bio import PDB
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from inverse_folding.data.transforms import DummyEncoding, ResidueNumberingEncoding
from inverse_folding.utils.constants import LETTER_TO_INDEX
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph


def pdb_to_graph(
    pdb: Path,
    antibody_chains: list[str],
    radius: float = 10.0,
    max_neighbors: int = 15,
    resnum_encoder: ResidueNumberingEncoding = DummyEncoding(),
) -> Data:
    # read structure from pdb
    parser = PDB.PDBParser(QUIET=True)
    complex_structure: Model = parser.get_structure(id=pdb.stem, file=str(pdb))[0]

    # iterate over residues and collect features
    ca_positions: list[Tensor] = []
    sequence: list[int] = []
    residue_numbers: list[int] = []

    for i, chain in enumerate(antibody_chains):
        for j, residue in enumerate(complex_structure[chain].get_residues()):
            residue = cast(Residue, residue)
            if residue.resname != "HOH":
                # collect coordinates
                ca_positions.append(torch.from_numpy(residue["CA"].coord))

                # encode residue
                residue_letter = protein_letters_3to1_extended.get(residue.resname, "X")
                sequence.append(LETTER_TO_INDEX.get(residue_letter, 20))

                # encode residue number
                residue_numbers.append(
                    resnum_encoder(
                        resnum=residue.id[1],
                        insertion_code=residue.id[2],
                        chain_position=j,
                        is_heavy_chain=i == 0,
                    )
                )

    # create graph based on distance between CA atoms
    x = torch.tensor(residue_numbers).long()
    positions = torch.stack(ca_positions, dim=0).float()
    edge_index = radius_graph(
        x=positions,
        r=radius,
        max_num_neighbors=max_neighbors,
    )
    row, col = edge_index
    edge_weight = (positions[row] - positions[col]).norm(dim=-1)

    return Data(
        x=x,
        pos=positions,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=torch.tensor(sequence).long(),
    )

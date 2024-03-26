# TODO: the module is not finished yet, just a draft

import os

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem

from ._utils import gen_fp, calc_sim


def mols_graph(mols_csv: str) -> nx.Graph:
    """Get a graph of molecules

    Parameters
    ----------
    mols_csv : str
        CSV file containing identifiers and SMILES of molecules

    Returns
    -------
    G : networkx.Graph
        A graph of molecules
    """

    df = pd.read_csv(mols_csv)
    G = nx.Graph()
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            G.add_node(row["title"], smiles=row["smiles"])
    for i, node1 in enumerate(G.nodes(data=True)):
        for j, node2 in enumerate(list(G.nodes(data=True))[i + 1 :]):
            mol1 = Chem.MolFromSmiles(node1[1]["smiles"])
            mol2 = Chem.MolFromSmiles(node2[1]["smiles"])
            fp1 = gen_fp(mol1)
            fp2 = gen_fp(mol2)
            similarity = calc_sim(fp1, fp2)
            if similarity > 0.8:
                G.add_edge(node1[0], node2[0])
    return G


def mols_subgraphs(G: nx.Graph, n: int) -> list[nx.Graph]:
    """Get a list of subgraphs of a graph

    Parameters
    ----------
    G : networkx.Graph
        A graph
    n : int
        Number of nodes in a subgraph

    Returns
    -------
    G_sub : list[networkx.Graph]
        A list of subgraphs
    """
    G_sub = [
        component
        for component in nx.connected_components(G)
        if len(component) == n or True
    ]

    return G_sub

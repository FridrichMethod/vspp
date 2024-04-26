import argparse
import logging
import multiprocessing as mp
import os
from typing import Any, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem

from vspp._utils import calc_bulk_sim, gen_fp

pandarallel.initialize(progress_bar=True)


def extract_query(
    df: pd.DataFrame,
    query: Chem.rdchem.Mol,
    smi_col: str = "smiles",
    *,
    cutoff: float = 0.8,
    upper_cutoff: float = 1.0,
    fp_type: str = "topological_torsion",
    similarity_metric: str = "dice",
) -> pd.DataFrame:
    """Extract similar structures

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    query : rdkit.Chem.rdchem.Mol
        Query structure
    smi_col : str, optional
        Column name for SMILES, by default "smiles"
    cutoff : float, optional
        Similarity cutoff, by default 0.8
    upper_cutoff : float, optional
        Upper cutoff for similarity, by default 1.0
    fp_type : str, optional
        Fingerprint type, by default "topological_torsion"
    similarity_metric : str, optional
        Similarity metric, by default "dice"

    Returns
    -------
    pandas.DataFrame
        A dataframe containing molecules similar to query.
    """

    if query is None:
        raise ValueError("No query structures are provided.")
    if df.empty:
        raise ValueError("No molecules are provided")

    query_fp = gen_fp(query, fp_type)
    df_copy = df.copy()

    # Generate fingerprints and calculate similarities
    logging.info("Generating fingerprints...")
    fps = df_copy[smi_col].parallel_apply(
        lambda x: gen_fp(Chem.MolFromSmiles(x), fp_type),
    )

    logging.info("Calculating similarities...")
    df_copy["similarity"] = calc_bulk_sim(query_fp, fps, similarity_metric)
    df_copy = df_copy[
        (df_copy["similarity"] > cutoff) & (df_copy["similarity"] <= upper_cutoff)
    ].reset_index(drop=True)

    if df_copy.empty:
        warn("No structures passed the similarity cutoff.")

    df_copy = df_copy.sort_values(
        by=["similarity", "title"], ascending=[False, True]
    ).reset_index(drop=True)

    logging.info("Similar structures are successfully extracted.")

    return df_copy


def extract_pattern(
    df: pd.DataFrame,
    pattern: Chem.rdchem.Mol,
    smi_col: str = "smiles",
) -> None:
    """Extract match structures

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    pattern : rdkit.Chem.rdchem.Mol
        Pattern structure
    smi_col : str, optional
        Column name for SMILES, by default "smiles"

    Returns
    -------
    None
    """

    if pattern is None:
        raise ValueError("No pattern structures are provided.")

    df_copy = df.copy()

    df_copy = df_copy[
        df_copy[smi_col].parallel_apply(
            lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(pattern)
        )
    ].reset_index(drop=True)

    if df_copy.empty:
        warn("No structures are matched.")

    df_copy = df_copy.sort_values(by="title").reset_index(drop=True)

    return df_copy


def main() -> None:
    """Main function"""


if __name__ == "__main__":
    main()

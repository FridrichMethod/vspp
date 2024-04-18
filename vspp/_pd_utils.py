import logging
from typing import Sequence
from warnings import warn

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

from vspp._utils import calc_bulk_sim, calc_descs, gen_fp, is_pains


def smi2pd(*args: str) -> pd.DataFrame:
    """Read .smi files to pandas.DataFrame

    Parameters
    ----------
    args : str
        Path to the .smi files

    Returns
    -------
    pandas.DataFrame
        A dataframe containing molecular structures

    Notes
    -----
    Only .smi files are read; other file formats are ignored.
    Return dataframe columns:
        - smiles: SMILES
        - title: Title
        - mol: RDKit molecule
    """

    df = pd.concat(
        [
            pd.read_csv(file, sep=" ", header=None, names=["smiles", "title"])
            for file in args
            if file.endswith(".smi")
        ],
        ignore_index=True,
    )

    logging.info("All .smi files are successfully loaded.")

    if df.empty:
        warn(
            "At least one non-empty .smi file should be provided; "
            "other file formats are ignored."
        )

    logging.info("Start generating molecular structures...")

    PandasTools.AddMoleculeColumnToFrame(
        df, smilesCol="smiles", molCol="mol", includeFingerprints=True
    )

    logging.info("Molecular structures are successfully generated.")

    df = df.dropna(subset="mol").reset_index(drop=True)

    df["smiles"] = df["mol"].apply(
        Chem.MolToSmiles, canonical=True, isomericSmiles=True
    )

    return df


def gen_info(df: pd.DataFrame) -> pd.DataFrame:
    """Generate molecular information

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures

    Returns
    -------
    pandas.DataFrame
        A dataframe containing molecular information

    Notes
    -----
    The following columns are added:
        - pains: PAINS alert
        - mol_wt: Molecular weight
        - log_p: LogP
        - h_acc: Hydrogen bond acceptor count
        - h_don: Hydrogen bond donor count
        - fsp3: Fraction of sp3 carbons
        - rot_bond: Rotatable bond count
        - ring_count: Ring count
        - tpsa: Topological polar surface area
        - druglikeness: Drug-likeness score
    """

    logging.info("Start generating molecular information...")

    if df.empty:
        warn("No molecular structures are provided.")
        return df

    df_copy = df.copy()

    df_copy["pains"] = df_copy["mol"].apply(is_pains)
    df_copy[
        [
            "mol_wt",
            "log_p",
            "h_acc",
            "h_don",
            "fsp3",
            "rot_bond",
            "ring_count",
            "tpsa",
            "druglikeness",
        ]
    ] = (
        df_copy["mol"].apply(calc_descs).apply(pd.Series)
    )

    logging.info("Molecular information is successfully generated.")

    return df_copy


def extract_by_queries(
    df: pd.DataFrame,
    queries: Sequence[Chem.rdchem.Mol],
    cutoff: float = 0.8,
    fp_type: str = "topological_torsion",
    similarity_metric: str = "dice",
) -> pd.DataFrame:
    """Extract similar structures

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    queries : Sequence[Chem.rdchem.Mol]
        Query structures
    cutoff : float, optional
        Similarity cutoff for extraction, by default 0.8
    fp_type : str, optional
        Fingerprint type for extraction, by default "topological_torsion"
    similarity_metric : str, optional
        Similarity metric for extraction, by default "dice"

    Returns
    -------
    pandas.DataFrame
        A dataframe containing extracted structures
    """

    if not (queries := [query for query in queries if query is not None]):
        raise ValueError("No query structures are provided.")

    df_copy = df.copy()
    queries_title = [query.GetProp("_Name") for query in queries]
    queries_fps = [gen_fp(query, fp_type) for query in queries]

    # Generate fingerprints and calculate similarities
    df_copy[queries_title] = (
        df_copy["mol"]
        .apply(gen_fp, fp_type=fp_type)
        .apply(calc_bulk_sim, args=(queries_fps, similarity_metric))
        .apply(pd.Series)
    )

    logging.info("Similarity calculation is successfully completed.")

    # Extract similar structures
    df_copy["query_title"] = df_copy[queries_title].idxmax(axis=1)
    df_copy["similarity"] = df_copy[queries_title].max(axis=1)
    df_copy = df_copy[df_copy["similarity"] >= cutoff].reset_index(drop=True)

    # Drop the query columns
    df_copy = df_copy.drop(columns=queries_title)

    if df_copy.empty:
        warn("No structures are extracted this time.")

    return df_copy


def extract_by_patterns(
    df: pd.DataFrame,
    patterns: Sequence[Chem.rdchem.Mol],
) -> pd.DataFrame:
    """Extract matched structures

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    patterns : Sequence[Chem.rdchem.Mol]
        Pattern structures

    Returns
    -------
    pandas.DataFrame
        A dataframe containing matched structures
    """

    if not (patterns := [pattern for pattern in patterns if pattern is not None]):
        raise ValueError("No pattern structures are provided.")

    df_copy = df.copy()
    patterns_smarts = [Chem.MolToSmarts(pattern) for pattern in patterns]

    # Substructure matching
    df_copy[patterns_smarts] = (
        df_copy["mol"].to_numpy() >= np.array(patterns).reshape(-1, 1)
    ).T

    logging.info("Substructure matching is successfully completed.")

    # Extract match structures
    df_copy = df_copy[df_copy[patterns_smarts].any(axis=1)].reset_index(drop=True)
    df_copy["pattern_smarts"] = df_copy[patterns_smarts].idxmax(axis=1)

    # Drop the pattern columns
    df_copy = df_copy.drop(columns=patterns_smarts)

    if df_copy.empty:
        warn("No structures are extracted this time.")

    return df_copy

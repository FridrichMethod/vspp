import argparse
import logging
import multiprocessing as mp
import os
from typing import Any, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from numba import float32, int64, vectorize
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import PandasTools

from vspp._utils import calc_bulk_sim, calc_descs, cluster_fps, gen_fp, is_pains

pandarallel.initialize(progress_bar=True)


def smi2df(file: str) -> pd.DataFrame:
    """Read .smi files to pandas.DataFrame

    Parameters
    ----------
    file : str
        Path to the .smi file

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
    """

    if not file.endswith(".smi"):
        raise ValueError("Only .smi files are supported.")

    df = pd.read_csv(file, sep=" ", header=None, names=["smiles", "title"])

    logging.info("Generating canonical SMILES...")
    df["smiles"] = df["smiles"].parallel_apply(
        lambda x: (
            np.nan
            if Chem.MolFromSmiles(x) is None
            else Chem.MolToSmiles(
                Chem.MolFromSmiles(x), canonical=True, isomericSmiles=True
            )
        )
    )

    logging.info("Dropping invalid SMILES and duplicates...")
    df = df.dropna(subset="smiles").reset_index(drop=True)
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)

    logging.info("Sorting SMILES by title...")
    df = df.sort_values(by="title").reset_index(drop=True)

    logging.info("The .smi file is successfully loaded.")

    return df


def extract_query(
    df: pd.DataFrame,
    query: Chem.rdchem.Mol,
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
    fps = df_copy["smiles"].parallel_apply(
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
) -> None:
    """Extract match structures

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    pattern : rdkit.Chem.rdchem.Mol
        Pattern structure

    Returns
    -------
    None
    """

    if pattern is None:
        raise ValueError("No pattern structures are provided.")

    df_copy = df.copy()

    df_copy = df_copy[
        df_copy["smiles"].parallel_apply(
            lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(pattern)
        )
    ].reset_index(drop=True)

    if df_copy.empty:
        warn("No structures are matched.")

    df_copy = df_copy.sort_values(by="title").reset_index(drop=True)

    return df_copy


def cluster_df(
    df: pd.DataFrame,
    *,
    cutoff: float = 0.6,
    fp_type: str = "morgan",
    similarity_metric: str = "tanimoto",
) -> pd.DataFrame:
    """Cluster match structures

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    cutoff : float, optional
        Tanimoto similarity cutoff, by default 0.6
    fp_type : str, optional
        Fingerprint type, by default "morgan"
    similarity_metric : str, optional
        Similarity metric, by default "tanimoto"

    Returns
    -------
    pandas.DataFrame
        A dataframe with cluster information
    """

    if df.empty:
        raise ValueError("No molecules are provided")

    df_copy = df.copy()

    logging.info("Generate fingerprints...")
    fps = np.vstack(
        df["smiles"].parallel_apply(
            lambda x: np.array(
                gen_fp(Chem.MolFromSmiles(x), fp_type=fp_type), dtype=bool
            )
        )
    )

    clusters = cluster_fps(fps, cutoff, similarity_metric, multiprocessing=True)  # type: ignore

    df_copy[["cluster_id", "cluster_size", "cluster_centroid"]] = (
        pd.DataFrame.from_dict(
            {
                idx: (i, len(cluster), j == 0)
                for i, cluster in enumerate(clusters)
                for j, idx in enumerate(cluster)
            },
            orient="index",
        )
    )

    df_copy = df_copy.sort_values(by=["cluster_id", "title"]).reset_index(drop=True)

    return df_copy


def gen_df_info(df: pd.DataFrame, *args) -> pd.DataFrame:
    """Generate molecular information

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    args
        Additional arguments for `calc_descs`

    Returns
    -------
    pandas.DataFrame
        A dataframe containing molecular information

    Notes
    -----
    The following columns will be added by default:
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

    if df.empty:
        raise ValueError("No molecules are provided")
    df_copy = df.copy()

    def _gen_info(smiles: str) -> pd.Series:
        mol = Chem.MolFromSmiles(smiles)
        return pd.Series(
            (
                is_pains(mol),
                *calc_descs(mol, *args),  # type: ignore
            )
        )

    logging.info("Start generating molecular information...")

    df_copy[
        [
            "pains",
            *(
                args
                or (
                    "mol_wt",
                    "log_p",
                    "h_acc",
                    "h_don",
                    "fsp3",
                    "rot_bond",
                    "ring_count",
                    "tpsa",
                    "druglikeness",
                )
            ),
        ]
    ] = df_copy["smiles"].parallel_apply(_gen_info)

    logging.info("Molecular information is successfully generated.")

    return df_copy


def write_df(
    df: pd.DataFrame,
    output_file: str,
    *,
    image_size: tuple = (300, 300),
) -> None:
    """Write the output files and draw structures

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    output_file : str
        Path to the output file
    image_size : tuple, optional
        Image size, by default (300, 300)

    Returns
    -------
    None
    """

    if df.empty:
        raise ValueError("No molecules are provided")

    if output_file.endswith(".xlsx"):
        PandasTools.AddMoleculeColumnToFrame(
            df, "smiles", "mol", includeFingerprints=True
        )
        logging.info("Molecular structures are added to dataframe.")
        PandasTools.SaveXlsxFromFrame(
            df,
            output_file,
            molCol="mol",
            size=image_size,
        )
        logging.info("Write %s.xlsx", os.path.basename(output_file))
    elif output_file.endswith(".csv"):
        df.to_csv(output_file, index=False)
        logging.info("Write %s.csv", os.path.basename(output_file))
    else:
        raise ValueError("Unsupported output format")


def main() -> None:
    """Main function"""


if __name__ == "__main__":
    main()

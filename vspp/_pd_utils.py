import logging
import multiprocessing as mp
import os
from typing import Any, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold

from vspp._utils import calc_descs, cluster_fps, gen_fp, is_pains

pandarallel.initialize(progress_bar=True)


def smi2df(
    file: str,
    *,
    sep: str = " ",
    no_sort: bool = False,
) -> pd.DataFrame:
    """Read .smi files to pandas.DataFrame

    Parameters
    ----------
    file : str
        Path to the .smi file
    sep : str, optional
        Separator, by default " "
    no_sort : bool, optional
        Preserve the original order, by default False

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

    df = pd.read_csv(file, sep=sep, header=None, names=["smiles", "title"])

    logging.info("Generating canonical SMILES...")
    df["smiles"] = df["smiles"].parallel_apply(
        lambda x: (
            np.nan if (mol := Chem.MolFromSmiles(x)) is None else Chem.MolToSmiles(mol)
        )
    )

    logging.info("Dropping invalid SMILES and duplicates...")
    len_0 = len(df)
    df = df.dropna(subset="smiles").reset_index(drop=True)
    len_1 = len(df)
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)
    len_2 = len(df)
    logging.info(
        "Dropped %d invalid SMILES and %d duplicates.", len_0 - len_1, len_1 - len_2
    )

    logging.info("Sorting SMILES by title...")
    if not no_sort:
        df = df.sort_values(by="title").reset_index(drop=True)

    logging.info("The .smi file is successfully loaded.")

    return df


def cluster_df_fps(
    df: pd.DataFrame,
    smi_col: str = "smiles",
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
    smi_col : str, optional
        Column name for SMILES, by default "smiles"
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
        df[smi_col].parallel_apply(
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


def cluster_df_frameworks(
    df: pd.DataFrame,
    smi_col: str = "smiles",
) -> pd.DataFrame:
    """Cluster frameworks

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    smi_col : str, optional
        Column name for SMILES, by default "smiles"

    Returns
    -------
    pandas.DataFrame
        A dataframe with cluster information
    """

    def _get_framework_with_johnson_sim(smiles: str) -> pd.DataFrame:
        """Get the Murcko scaffold of a molecule and calculate the Johnson similarity"""
        mol = Chem.MolFromSmiles(smiles)
        core = MurckoScaffold.GetScaffoldForMol(mol)

        # `core` may be an empty molecule if no rings are present, but not None
        # if Chem.MolToSmiles(core):
        #     side_chains = Chem.ReplaceCore(mol, core)

        similarity = (core.GetNumAtoms() + core.GetNumBonds()) / (
            mol.GetNumAtoms() + mol.GetNumBonds()
        )

        return pd.Series((Chem.MolToSmiles(core), similarity))

    if df.empty:
        raise ValueError("No molecules are provided")
    df_copy = df.copy()

    logging.info("Generating frameworks...")

    df_copy[["cluster_framework", "cluster_similarity"]] = df_copy[
        smi_col
    ].parallel_apply(_get_framework_with_johnson_sim)

    logging.info("Molecules are clustered successfully!")

    df_copy["cluster_size"] = df_copy.groupby("cluster_framework")[
        "cluster_framework"
    ].transform("size")
    df_copy = df_copy.sort_values(
        by=["cluster_size", "cluster_framework", "cluster_similarity"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    df_copy = pd.merge(
        df_copy,
        df_copy["cluster_framework"]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "cluster_id"}),
        how="left",
        on="cluster_framework",
    )

    return df_copy


def gen_df_info(
    df: pd.DataFrame,
    *args,
    smi_col: str = "smiles",
) -> pd.DataFrame:
    """Generate molecular information

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing molecular structures
    args
        Additional arguments for `calc_descs`
    smi_col : str, optional
        Column name for SMILES, by default "smiles"

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
    ] = df_copy[smi_col].parallel_apply(_gen_info)

    logging.info("Molecular information is successfully generated.")

    return df_copy


def write_df(
    df: pd.DataFrame,
    output_file: str,
    smi_col: str = "smiles",
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
    smi_col : str, optional
        Column name for SMILES, by default "smiles"
    image_size : tuple, optional
        Image size, by default (300, 300)

    Returns
    -------
    None
    """

    if df.empty:
        raise ValueError("No molecules are provided")

    if output_file.endswith(".xlsx"):
        df_copy = df.copy()
        # Convert boolean columns to integer
        # Note: PandasTools.SaveXlsxFromFrame does not support boolean columns
        for col in df_copy.columns:
            if df_copy[col].dtype == bool:
                df_copy[col] = df_copy[col].astype(int)
        PandasTools.AddMoleculeColumnToFrame(
            df_copy, smi_col, "mol", includeFingerprints=True
        )
        logging.info("Molecular structures are added to dataframe.")
        PandasTools.SaveXlsxFromFrame(
            df_copy,
            output_file,
            molCol="mol",
            size=image_size,
        )
        logging.info("Write %s", os.path.basename(output_file))
    elif output_file.endswith(".csv"):
        df.to_csv(output_file, index=False)
        logging.info("Write %s", os.path.basename(output_file))
    else:
        raise ValueError("Unsupported output format")

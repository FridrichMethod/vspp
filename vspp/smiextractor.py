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
from rdkit.Chem import PandasTools

from vspp._utils import (
    calc_descs,
    calc_sim,
    cluster_fps,
    draw_mols,
    gen_fp,
    is_pains,
    smart_tqdm,
)

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


def gen_info(df: pd.DataFrame, *args) -> pd.DataFrame:
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
        raise ValueError("No molecular structures are found in the dataframe.")
    df_copy = df.copy()

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
    ] = df_copy["smiles"].parallel_apply(
        lambda x: pd.Series(
            (is_pains(Chem.MolFromSmiles(x)), *calc_descs(Chem.MolFromSmiles(x, *args)))
        )
    )

    logging.info("Molecular information is successfully generated.")

    return df_copy


def write_df(
    df: pd.DataFrame,
    output_dir: str,
    file_name: str,
    xlsx: bool = True,
    image_size: tuple = (300, 300),
) -> None:
    """Write the output files and draw structures

    Parameters
    ----------
    output_dir : str
        Path to the output directory
    file_name : str
        Output file name
    xlsx : bool, optional
        Whether to write an Excel file, by default True
    image_size : tuple, optional
        Image size, by default (300, 300)
    """

    if df.empty:
        raise ValueError("No similar structures are extracted yet.")

    os.makedirs(output_dir, exist_ok=True)
    if xlsx:
        PandasTools.AddMoleculeColumnToFrame(
            df, "smiles", "mol", includeFingerprints=True
        )
        PandasTools.SaveXlsxFromFrame(
            df,
            os.path.join(output_dir, f"{file_name}.xlsx"),
            molCol="mol",
            size=image_size,
        )
        logging.info("Write %s.xlsx", file_name)
    else:
        df.to_csv(os.path.join(output_dir, f"{file_name}.csv"), index=False)
        logging.info("Write %s.csv", file_name)


class SimExtractor:
    """A class to extract similar structures from a compound library"""

    def __init__(
        self,
        query: Chem.rdchem.Mol,
        *,
        cutoff: float = 0.8,
        upper_cutoff: float = 1.0,
        fp_type: str = "topological_torsion",
        similarity_metric: str = "dice",
    ) -> None:
        """Initialize the SimExtractor

        Parameters
        ----------
        query : rdkit.Chem.rdchem.Mol
            A query molecule
        cutoff : float, optional
            Dice similarity cutoff, by default 0.8
        upper_cutoff : float, optional
            Upper cutoff for similarity, by default 1.0
        fp_type : str, optional
            Fingerprint type, by default "topological_torsion"
        similarity_metric : str, optional
            Similarity metric, by default "dice"

        Returns
        -------
        None
        """

        if query is None:
            raise ValueError("The query molecule is invalid.")
        self.query = query
        self.query_title = query.GetProp("_Name")
        self.query_fp = gen_fp(query, fp_type)

        self.cutoff = cutoff
        self.upper_cutoff = upper_cutoff
        self.fp_type = fp_type
        self.similarity_metric = similarity_metric

        self.structs: pd.DataFrame = pd.DataFrame()

    def extract(
        self,
        file: str,
    ) -> None:
        """Extract similar structures

        Parameters
        ----------
        file : str
            Path to the .smi file

        Returns
        -------
        None
        """

        if self.query is None:
            raise ValueError("No query structures are provided.")
        query_fp = gen_fp(self.query, self.fp_type)

        df = smi2df(file)

        # Generate fingerprints and calculate similarities
        df["similarity"] = df["smiles"].parallel_apply(
            lambda x: calc_sim(
                gen_fp(Chem.MolFromSmiles(x), self.fp_type),
                query_fp,
                self.similarity_metric,
            )
        )
        logging.info("Similarity calculation is successfully completed.")

        df = df[
            (df["similarity"] > self.cutoff) & (df["similarity"] <= self.upper_cutoff)
        ].reset_index(drop=True)

        if df.empty:
            warn("No structures passed the similarity cutoff.")

        df = gen_info(df)

        self.structs = df

    def write(
        self,
        output_dir: str,
        xlsx: bool = True,
        image_size: tuple = (300, 300),
    ) -> None:
        """Write the output files and draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        xlsx : bool, optional
            Whether to write an Excel file, by default True
        image_size : tuple, optional
            Image size, by default (300, 300)
        """

        write_df(
            self.structs,
            output_dir,
            self.query_title,
            xlsx=xlsx,
            image_size=image_size,
        )

    def draw(self, output_dir: str, **kwargs) -> None:
        """Draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        kwargs
            Other keyword arguments for `draw_mols`.
        """

        if self.structs.empty:
            raise ValueError("No similar structures are extracted yet.")

        os.makedirs(output_dir, exist_ok=True)
        draw_mols(
            self.structs["smiles"].apply(Chem.MolFromSmiles).to_list(),
            os.path.join(output_dir, f"{self.query_title}.png"),
            legends=[
                " ".join(x)
                for x in zip(
                    self.structs["title"],
                    self.structs["similarity"].apply(lambda x: f"{x:.3f}"),
                )
            ],
            **kwargs,
        )
        logging.info("Draw %s.png", self.query_title)


class MatExtractor:
    """A class to extract match structures from a compound library"""

    def __init__(
        self,
        pattern: Chem.rdchem.Mol,
    ) -> None:
        """Initialize the MatExtractor

        Parameters
        ----------
        pattern : rdkit.Chem.rdchem.Mol
            A SMARTS pattern

        Returns
        -------
        None
        """

        if pattern is None:
            raise ValueError("The pattern is invalid.")
        self.pattern = pattern
        self.smarts = Chem.MolToSmarts(pattern, isomericSmiles=True)

        self.structs: pd.DataFrame = pd.DataFrame()

    def extract(
        self,
        file: str,
    ) -> None:
        """Extract matched structures

        Parameters
        ----------
        file : str
            Path to the .smi file

        Returns
        -------
        None
        """

        if self.pattern is None:
            raise ValueError("No pattern structures are provided.")

        df = smi2df(file)

        df = df[
            df["smiles"].parallel_apply(
                lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(self.pattern)
            )
        ].reset_index(drop=True)

        if df.empty:
            warn("No structures are matched.")

        df = gen_info(df)

        self.structs = df

    def cluster(
        self,
        *,
        cutoff: float = 0.6,
        fp_type: str = "morgan",
        similarity_metric: str = "tanimoto",
    ) -> None:
        """Cluster match structures

        Parameters
        ----------
        cutoff : float, optional
            Tanimoto similarity cutoff, by default 0.6
        fp_type : str, optional
            Fingerprint type, by default "morgan"
        similarity_metric : str, optional
            Similarity metric, by default "tanimoto"

        Returns
        -------
        None
        """

        if self.structs.empty:
            raise ValueError("No match structures are extracted yet.")

        logging.info("Generate fingerprints...")
        fps = (
            self.structs["smiles"]
            .parallel_apply(lambda x: gen_fp(Chem.MolFromSmiles(x), fp_type))
            .to_list()
        )

        clusters = cluster_fps(fps, cutoff, similarity_metric)

        df = pd.DataFrame.from_dict(
            {
                idx: (i, len(cluster), j == 0)
                for i, cluster in enumerate(clusters)
                for j, idx in enumerate(cluster)
            },
            orient="index",
        )

        self.structs[["cluster_id", "cluster_size", "cluster_centroid"]] = df

    def write(
        self,
        output_dir: str,
        xlsx: bool = True,
        image_size: tuple = (300, 300),
    ) -> None:
        """Write the output files and draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        xlsx : bool, optional
            Whether to write an Excel file, by default True
        image_size : tuple, optional
            Image size, by default (300, 300)
        """

        write_df(
            self.structs,
            output_dir,
            self.smarts,
            xlsx=xlsx,
            image_size=image_size,
        )

    def draw(self, output_dir: str, centroid: bool = True, **kwargs) -> None:
        """Draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        centroid : bool, optional
            Whether only to draw the cluster centroids, by default True
        kwargs
            Other keyword arguments for `draw_mols`.
        """

        if self.structs.empty:
            raise ValueError("No similar structures are extracted yet.")
        if "cluster_centroid" not in self.structs.columns:
            raise ValueError("No clusters are generated yet.")

        os.makedirs(output_dir, exist_ok=True)
        if centroid:
            draw_mols(
                self.structs[self.structs["cluster_centroid"]]["smiles"]
                .apply(Chem.MolFromSmiles)
                .to_list(),
                os.path.join(output_dir, f"{self.smarts}.png"),
                legends=self.structs[self.structs["cluster_centroid"]][
                    "title"
                ].to_list(),
                pattern=self.pattern,
                **kwargs,
            )
        else:
            draw_mols(
                self.structs["smiles"].apply(Chem.MolFromSmiles).to_list(),
                os.path.join(output_dir, f"{self.smarts}.png"),
                legends=self.structs["title"].to_list(),
                pattern=self.pattern,
                **kwargs,
            )
        logging.info("Draw %s.png", self.smarts)

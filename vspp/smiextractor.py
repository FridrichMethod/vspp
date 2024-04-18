import argparse
import logging
import multiprocessing as mp
import os
from typing import Any

import pandas as pd
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

    def _compare_molecules(
        self,
        line: str,
    ) -> list[Any]:
        """Compare a query with a line of SMILES and title in a .smi file

        Parameters
        ----------
        line : str
            A line of SMILES and title, separated by a space

        Returns
        -------
        struct_info : list[list[Any]]
            A similar structure and its properties if any.
        """

        mol = Chem.MolFromSmiles(line)

        # It is possible that SMILES Parse Error occurs (by warning)
        # if the SMILES is invalid (possibly due to the header in the .smi file)
        if mol is None:
            return []
        fp = gen_fp(mol, self.fp_type)

        if (
            self.upper_cutoff
            > (similarity := calc_sim(fp, self.query_fp, self.similarity_metric))
            > self.cutoff
        ):
            return [
                mol.GetProp("_Name"),
                Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                similarity,
                is_pains(mol),
                *calc_descs(mol),  # type: ignore
            ]

        return []

    def extract(
        self,
        file: str,
    ) -> None:
        """Extract similar structures from a compound library

        Parameters
        ----------
        file : str
            Path to the compound file, should be a .smi file, with or without a header.
            Module `vspp.smiconverter` could be used to convert files to .smi files.

        Returns
        -------
        None
        """

        # Use `line`, `query_title` and `query_fps`
        # instead of `query: rdkit.Chem.rdchem.Mol` to save memory
        with open(file, "r", encoding="utf-8") as f:
            _args = f.readlines()

        # Compare molecules in parallel
        logging.info("Start comparing molecules...")
        with mp.Pool(mp.cpu_count()) as pool:
            data = pool.map(
                self._compare_molecules,
                smart_tqdm(
                    _args,
                    total=len(_args),
                    desc="Comparing",
                    unit="mol",
                ),
            )
        logging.info("Molecules are extracted!")

        # Write the output file
        df = pd.DataFrame(
            data,
            columns=[
                "title",
                "smiles",
                "similarity",
                "pains",
                "mol_wt",
                "log_p",
                "h_acc",
                "h_don",
                "fsp3",
                "rot_bond",
                "ring_count",
                "tpsa",
                "druglikeness",
            ],
        ).dropna()

        self.structs = df.sort_values(
            by=["similarity", "title"],
            ascending=[False, True],
        ).reset_index(drop=True)

    def write(self, output_dir: str, xlsx: bool = True) -> None:
        """Write the output files and draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        xlsx : bool, optional
            Whether to write an Excel file, by default True
        """

        if self.structs.empty:
            raise ValueError("No similar structures are extracted yet.")

        os.makedirs(output_dir, exist_ok=True)
        if xlsx:
            PandasTools.AddMoleculeColumnToFrame(
                self.structs, "smiles", "mol", includeFingerprints=True
            )
            PandasTools.SaveXlsxFromFrame(
                self.structs,
                os.path.join(output_dir, f"{self.query_title}.xlsx"),
                molCol="mol",
                size=(300, 300),
            )
            logging.info("Write %s.xlsx", self.query_title)
        else:
            self.structs.to_csv(
                os.path.join(output_dir, f"{self.query_title}.csv"), index=False
            )
            logging.info("Write %s.csv", self.query_title)

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

    def _compare_molecules(
        self,
        line: str,
    ) -> list[Any]:
        """Compare a pattern with a line of SMILES and title in a .smi file

        Parameters
        ----------
        line : str
            A line of SMILES and title, separated by a space

        Returns
        -------
        struct_info : list[Any]
            A match structure and its properties if any.
        """

        mol = Chem.MolFromSmiles(line)

        # It is possible that SMILES Parse Error occurs (by warning)
        # if the SMILES is invalid (possibly due to the header in the .smi file)
        if mol is None:
            return []

        if mol.HasSubstructMatch(self.pattern):
            return [
                mol.GetProp("_Name"),
                Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                is_pains(mol),
                *calc_descs(mol),  # type: ignore
            ]

        return []

    def extract(
        self,
        file: str,
    ) -> None:
        """Extract match structures from a compound library

        Parameters
        ----------
        file : str
            Path to the compound file, should be a .smi file, with or without a header.
            Module `vspp.smiconverter` could be used to convert files to .smi files.

        Returns
        -------
        None
        """

        # Use `line`, `query_title` and `query_fps`
        # instead of `query: rdkit.Chem.rdchem.Mol` to save memory
        with open(file, "r", encoding="utf-8") as f:
            _args = f.readlines()

        # Compare molecules in parallel
        logging.info("Start comparing molecules...")
        with mp.Pool(mp.cpu_count()) as pool:
            data = pool.map(
                self._compare_molecules,
                smart_tqdm(
                    _args,
                    total=len(_args),
                    desc="Comparing",
                    unit="mol",
                ),
            )
        logging.info("Molecules are extracted!")

        # Write the output file
        df = pd.DataFrame(
            data,
            columns=[
                "title",
                "smiles",
                "pains",
                "mol_wt",
                "log_p",
                "h_acc",
                "h_don",
                "fsp3",
                "rot_bond",
                "ring_count",
                "tpsa",
                "druglikeness",
            ],
        ).dropna()

        self.structs = df.sort_values(by="title").reset_index(drop=True)

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
            .apply(lambda x: gen_fp(Chem.MolFromSmiles(x), fp_type))
            .to_list()
        )

        clusters = cluster_fps(fps, cutoff, similarity_metric)

        columns = ["cluster_id", "cluster_size", "cluster_centroid"]

        df = pd.DataFrame.from_dict(
            {
                idx: (i, len(cluster), j == 0)
                for i, cluster in enumerate(clusters)
                for j, idx in enumerate(cluster)
            },
            orient="index",
            columns=columns,
        )

        self.structs[columns] = df

    def write(self, output_dir: str, xlsx: bool = True) -> None:
        """Write the output files and draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        xlsx : bool, optional
            Whether to write an Excel file, by default True
        """

        if self.structs.empty:
            raise ValueError("No similar structures are extracted yet.")

        os.makedirs(output_dir, exist_ok=True)
        if xlsx:
            PandasTools.AddMoleculeColumnToFrame(
                self.structs, "smiles", "mol", includeFingerprints=True
            )
            PandasTools.SaveXlsxFromFrame(
                self.structs,
                os.path.join(output_dir, f"{self.smarts}.xlsx"),
                molCol="mol",
                size=(300, 300),
            )
            logging.info("Write %s.xlsx", self.smarts)
        else:
            self.structs.to_csv(
                os.path.join(output_dir, f"{self.smarts}.csv"), index=False
            )
            logging.info("Write %s.csv", self.smarts)

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


def main() -> None: ...


if __name__ == "__main__":
    main()

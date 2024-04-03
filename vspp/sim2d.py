import argparse
import logging
import multiprocessing as mp
import os
from itertools import chain
from typing import Any, Sequence, Unpack
from warnings import warn

import pandas as pd
from rdkit import Chem, DataStructs

from vspp._utils import (
    calc_descs,
    calc_sim,
    draw_structures,
    gen_fp,
    is_pains,
    read_mols,
    smart_tqdm,
)


class SimExtractor:
    """A class to extract similar structures from a compound library"""

    def __init__(
        self,
        queries: Sequence[Chem.rdchem.Mol],
        *,
        cutoff: float = 0.8,
        fp_type: str = "topological_torsion",
        similarity_metric: str = "dice",
    ) -> None:
        """Initialize the SimExtractor

        Parameters
        ----------
        queries : Sequence[rdkit.Chem.rdchem.Mol]
            A list of query molecules
        cutoff : float, optional
            Dice similarity cutoff, by default 0.8
        fp_type : str, optional
            Fingerprint type, by default "topological_torsion"
        similarity_metric : str, optional
            Similarity metric, by default "dice"

        Returns
        -------
        None
        """

        self.queries = [mol for mol in queries if mol is not None]
        self.queries_title = [mol.GetProp("_Name") for mol in self.queries]
        self.queries_fps = [gen_fp(mol, fp_type) for mol in self.queries]

        self._cutoff = cutoff
        self._fp_type = fp_type
        self._similarity_metric = similarity_metric

        self.similar_structures: pd.DataFrame = pd.DataFrame()

    def __repr__(self) -> str:
        return (
            f"SimExtractor(queries={self.queries_title}, "
            f"cutoff={self._cutoff}, "
            f"fp_type={self._fp_type}, "
            f"similarity_metric={self._similarity_metric})"
        )

    def __len__(self) -> int:
        return len(self.similar_structures)

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float) -> None:
        self._cutoff = value
        self.similar_structures = pd.DataFrame()
        warn(
            "The similar structures are reset due to the change of cutoff."
            "Please run `self.extract` again to extract similar structures."
        )

    @property
    def fp_type(self) -> str:
        return self._fp_type

    @fp_type.setter
    def fp_type(self, value: str) -> None:
        self._fp_type = value
        self.similar_structures = pd.DataFrame()
        warn(
            "The similar structures are reset due to the change of fingerprint type."
            "Please run `self.extract` again to extract similar structures."
        )

    @property
    def similarity_metric(self) -> str:
        return self._similarity_metric

    @similarity_metric.setter
    def similarity_metric(self, value: str) -> None:
        self._similarity_metric = value
        self.similar_structures = pd.DataFrame()
        warn(
            "The similar structures are reset due to the change of similarity metric."
            "Please run `self.extract` again to extract similar structures."
        )

    def _compare_molecules(
        self,
        line: str,
    ) -> list[list[Any]]:
        """Compare queries with a line of SMILES and title in a .smi file

        Parameters
        ----------
        line : str
            A line of SMILES and title, separated by a space

        Returns
        -------
        sim_struct_info : list[list[Any]]
            A list of list of similar structures and their properties if any.
        """

        try:
            smiles, title = line.split()
        except ValueError:
            return []

        mol = Chem.MolFromSmiles(smiles)
        # It is possible that SMILES Parse Error occurs (by warning)
        # if the SMILES is invalid (possibly due to the header in the .smi file)
        if mol is None:
            return []
        fp = gen_fp(mol, self._fp_type)

        sim_struct_info = []
        for query_title, query_fp in zip(self.queries_title, self.queries_fps):
            similarity = calc_sim(fp, query_fp, self._similarity_metric)
            if similarity > self._cutoff:
                canonical_smiles = Chem.MolToSmiles(
                    mol, canonical=True, isomericSmiles=True
                )
                pains = is_pains(mol)
                descriptors = calc_descs(mol)
                sim_struct_info.append(
                    [
                        title,
                        canonical_smiles,
                        pains,
                        *descriptors,  # type: ignore
                        query_title,
                        similarity,
                    ]
                )

        return sim_struct_info

    def extract(
        self,
        compounds_file: str,
    ) -> pd.DataFrame:
        """Extract similar structures from a compound library

        Parameters
        ----------
        compounds_file : str
            Path to the compound file, should be a .smi file, with or without a header.
            Module `vspp.smiconverter` could be used to convert files to .smi files.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of similar structures.
        """

        # Use `line`, `queries_title` and `queries_fps`
        # instead of `queries: list[rdkit.Chem.rdchem.Mol]` to save memory
        with open(compounds_file, "r", encoding="utf-8") as f:
            _args = f.readlines()

        # Compare molecules in parallel
        logging.info("Start comparing molecules...")
        with mp.Pool(mp.cpu_count() - 1) as pool:
            raw_data = pool.map(
                self._compare_molecules,
                smart_tqdm(
                    _args,
                    total=len(_args),
                    desc="Comparing",
                    unit="mol",
                ),
            )

        # Flatten the list
        data = list(chain.from_iterable(raw_data))

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
                "query_title",
                "similarity",
            ],
        )
        df = df.sort_values(
            by=["query_title", "similarity", "title"],
            ascending=[True, False, True],
        )

        self.similar_structures = df

        return df

    def write(self, output_dir: str) -> None:
        """Write the output files and draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        """

        if self.similar_structures.empty:
            raise ValueError("No similar structures are extracted yet.")

        os.makedirs(output_dir, exist_ok=True)
        for query_title, group in self.similar_structures.groupby("query_title"):
            group.to_csv(os.path.join(output_dir, f"{query_title}.csv"), index=False)
            logging.info("Write %s.csv", query_title)

    def draw(self, output_dir: str, **kwargs) -> None:
        """Draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        kwargs :
            Other keyword arguments for `draw_structures`.
        """

        if self.similar_structures.empty:
            raise ValueError("No similar structures are extracted yet.")

        for query_title, group in self.similar_structures.groupby("query_title"):
            draw_structures(
                group["smiles"].apply(Chem.MolFromSmiles).tolist(),
                os.path.join(output_dir, f"{query_title}.png"),
                titles=(
                    group["title"].tolist(),
                    group["query_title"].tolist(),
                    group["similarity"].apply(lambda x: f"{x:.3f}").tolist(),
                ),
                **kwargs,
            )
            logging.info("Draw %s.png", query_title)


def extract_similar_structures(
    queries_file: str,
    compounds_file: str,
    output_dir: str | None = None,
    *,
    cutoff: float = 0.8,
    fp_type: str = "topological_torsion",
    similarity_metric: str = "dice",
    **kwargs,
) -> pd.DataFrame:
    """Extract similar structures from a compound library

    Parameters
    ----------
    queries_file : str
        Path to the query file
    compounds_file : str
        Path to the compound file, should be a .smi file, with or without a header.
        Module `vspp.smiconverter` could be used to convert files to .smi files.
    output_dir : str, optional
        Path to the output directory, by default None
    cutoff : float, optional
        Dice similarity cutoff, by default 0.8
    fp_type : str, optional
        Fingerprint type, by default "topological_torsion"
    similarity_metric : str, optional
        Similarity metric, by default "dice"

    Returns
    -------
    df : pandas.DataFrame
        A dataframe of similar structures.
    """

    # Read the queries file
    queries = [mol for mol in read_mols(queries_file) if mol is not None]

    # Read the compound file
    if not compounds_file.endswith(".smi"):
        raise ValueError(
            "The compound file should be a .smi file with a header. "
            "Module `vspp.smiconverter` could be used to convert files to .smi files."
        )

    # Initialize the SimExtractor
    sim_extractor = SimExtractor(
        queries,
        cutoff=cutoff,
        fp_type=fp_type,
        similarity_metric=similarity_metric,
    )

    # Compare molecules in parallel
    sim_extractor.extract(compounds_file)

    # Write the output file
    if output_dir is not None:
        sim_extractor.write(output_dir)
    # Draw structures
    if output_dir is not None:
        sim_extractor.draw(output_dir, **kwargs)

    return sim_extractor.similar_structures


class MatExtractor:
    """A class to extract match structures from a compound library"""

    def __init__(
        self,
        patterns: Sequence[Chem.rdchem.Mol],
    ) -> None:
        """Initialize the MatExtractor

        Parameters
        ----------
        patterns : Sequence[rdkit.Chem.rdchem.Mol]
            A list of SMARTS patterns

        Returns
        -------
        None
        """

        self.patterns = [pattern for pattern in patterns if pattern is not None]
        self.patterns_smarts = [Chem.MolToSmarts(pattern) for pattern in self.patterns]

        self.match_structures: pd.DataFrame = pd.DataFrame()

    def __repr__(self) -> str:
        return f"MatExtractor(patterns={self.patterns_smarts})"

    def __len__(self) -> int:
        return len(self.match_structures)

    def _compare_molecules(
        self,
        line: str,
    ) -> list[list[Any]]:
        """Compare patterns with a line of SMILES and title in a .smi file

        Parameters
        ----------
        line : str
            A line of SMILES and title, separated by a space

        Returns
        -------
        match_struct_info : list[list[Any]]
            A list of list of match structures and their properties if any.
        """

        try:
            smiles, title = line.split()
        except ValueError:
            return []

        mol = Chem.MolFromSmiles(smiles)
        # It is possible that SMILES Parse Error occurs (by warning)
        # if the SMILES is invalid (possibly due to the header in the .smi file)
        if mol is None:
            return []

        mat_struct_info = []
        for pattern, smarts in zip(self.patterns, self.patterns_smarts):
            if mol.HasSubstructMatch(pattern):
                canonical_smiles = Chem.MolToSmiles(
                    mol, canonical=True, isomericSmiles=True
                )
                pains = is_pains(mol)
                descriptors = calc_descs(mol)
                mat_struct_info.append(
                    [
                        title,
                        canonical_smiles,
                        pains,
                        *descriptors,  # type: ignore
                        smarts,
                    ]
                )

        return mat_struct_info

    def extract(
        self,
        compounds_file: str,
    ) -> pd.DataFrame:
        """Extract match structures from a compound library

        Parameters
        ----------
        compounds_file : str
            Path to the compound file, should be a .smi file, with or without a header.
            Module `vspp.smiconverter` could be used to convert files to .smi files.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of match structures.
        """

        # Use `line` and `patterns_title`
        # instead of `patterns: list[rdkit.Chem.rdchem.Mol]` to save memory
        with open(compounds_file, "r", encoding="utf-8") as f:
            _args = f.readlines()

        # Compare molecules in parallel
        logging.info("Start comparing molecules...")
        with mp.Pool(mp.cpu_count() - 1) as pool:
            raw_data = pool.map(
                self._compare_molecules,
                smart_tqdm(
                    _args,
                    total=len(_args),
                    desc="Comparing",
                    unit="mol",
                ),
            )

        # Flatten the list
        data = list(chain.from_iterable(raw_data))

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
                "smarts",
            ],
        )
        df = df.sort_values(
            by=["smarts", "title"],
            ascending=[True, True],
        )

        self.match_structures = df

        return df

    def write(self, output_dir: str) -> None:
        """Write the output files and draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        """

        if self.match_structures.empty:
            raise ValueError("No match structures are extracted yet.")

        os.makedirs(output_dir, exist_ok=True)
        for smarts, group in self.match_structures.groupby("smarts"):
            group.to_csv(os.path.join(output_dir, f"{smarts}.csv"), index=False)
            logging.info("Write %s.csv", smarts)

    def draw(self, output_dir: str, **kwargs) -> None:
        """Draw structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        kwargs :
            Other keyword arguments for `draw_structures`.
        """

        if self.match_structures.empty:
            raise ValueError("No match structures are extracted yet.")

        for smarts, group in self.match_structures.groupby("smarts"):
            draw_structures(
                group["smiles"].apply(Chem.MolFromSmiles).tolist(),
                os.path.join(output_dir, f"{smarts}.png"),
                titles=group["title"].tolist(),
                pattern=Chem.MolFromSmarts(smarts),
                **kwargs,
            )
            logging.info("Draw %s.png", smarts)


def extract_match_structures(
    patterns_smarts: list[str],
    compounds_file: str,
    output_dir: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Extract match structures from a compound library

    Parameters
    ----------
    patterns_smarts : list[str]
        A list of SMARTS patterns
    compounds_file : str
        Path to the compound file, should be a .smi file, with or without a header.
        Module `vspp.smiconverter` could be used to convert files to .smi files.
    output_dir : str, optional
        Path to the output directory, by default None

    Returns
    -------
    df : pandas.DataFrame
        A dataframe of match structures.
    """

    # Read the patterns list
    patterns = [
        pattern for smarts in patterns_smarts if (pattern := Chem.MolFromSmarts(smarts)) is not None
    ]

    # Read the compound file
    if not compounds_file.endswith(".smi"):
        raise ValueError(
            "The compound file should be a .smi file with a header. "
            "Module `vspp.smiconverter` could be used to convert files to .smi files."
        )

    # Initialize the MatExtractor
    mat_extractor = MatExtractor(patterns)

    # Compare molecules in parallel
    mat_extractor.extract(compounds_file)

    # Write the output file
    if output_dir is not None:
        mat_extractor.write(output_dir)
    # Draw structures
    if output_dir is not None:
        mat_extractor.draw(output_dir, **kwargs)

    return mat_extractor.match_structures


def main() -> None:
    """Main function

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--queries",
        type=str,
        help="Path to the query file",
    )
    parser.add_argument(
        "-t",
        "--patterns",
        type=str,
        nargs="+",
        help="A list of SMARTS patterns",
    )
    parser.add_argument(
        "-p",
        "--compounds",
        type=str,
        required=True,
        help="Path to the compound file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=0.8,
        help="Similarity cutoff, by default 0.8",
    )
    parser.add_argument(
        "-f",
        "--fp_type",
        type=str,
        default="topological_torsion",
        help="Fingerprint type, by default 'topological_torsion'",
    )
    parser.add_argument(
        "-s",
        "--similarity_metric",
        type=str,
        default="dice",
        help="Similarity metric, by default 'dice'",
    )

    args = parser.parse_args()
    
    if args.queries is not None:
        extract_similar_structures(
            args.queries,
            args.compounds,
            args.output,
            cutoff=args.cutoff,
            fp_type=args.fp_type,
            similarity_metric=args.similarity_metric,
        )
    
    if args.patterns is not None:
        extract_match_structures(
            args.patterns,
            args.compounds,
            args.output,
        )


if __name__ == "__main__":
    main()

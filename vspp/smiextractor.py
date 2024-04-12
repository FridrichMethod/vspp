import argparse
import logging
import os
from typing import Any, Literal, Sequence, Unpack
from warnings import warn

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

from vspp._utils import (
    MolSupplier,
    calc_bulk_sim,
    calc_descs,
    cluster_fps,
    draw_mols,
    gen_fp,
    is_pains,
)


class SmiExtractor:
    """Extract similar and matched structures from .smi files

    Attributes
    ----------
    structs : pandas.DataFrame
        A dataframe containing molecular structures

    Methods
    -------
    read_smi(*args: str) -> pandas.DataFrame
        Read .smi files
    gen_info(df: pandas.DataFrame) -> pandas.DataFrame
        Generate molecular information
    extract(
        *args: str,
        queries: Sequence[Chem.rdchem.Mol] | None = None,
        patterns: Sequence[Chem.rdchem.Mol] | None = None,
        cutoff: float = 0.8,
        fp_type: str = "topological_torsion",
        similarity_metric: str = "dice",
    ) -> None
        Extract similar and matched structures
    deduplicate() -> None
        Deduplicate the structures
    sort() -> None
        Sort the structures
    cluster(
        cutoff: float = 0.6,
        fp_type: str = "morgan",
        similarity_metric: str = "tanimoto",
    ) -> None
        Cluster the structures
    write(
        output_dir: str,
        by: Literal["structs", "query", "pattern"] = "structs",
        xlsx: bool = False,
    ) -> None
        Write the output files
    draw(
        output_dir: str,
        by: Literal["structs", "query", "pattern"] = "structs",
        centroid: bool = False,
        **kwargs: Any,
    ) -> None
        Draw 2D structures
    """

    def __init__(self) -> None:
        """Initialize the SmiExtractor object"""

        self.structs: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def read_smi(*args: str) -> pd.DataFrame:
        """Read .smi files

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

    @staticmethod
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

        df["pains"] = df["mol"].apply(is_pains)
        df[
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
            df["mol"].apply(calc_descs).apply(pd.Series)
        )

        logging.info("Molecular information is successfully generated.")

        return df

    def extract(
        self,
        *args: str,
        queries: Sequence[Chem.rdchem.Mol] = (),
        patterns: Sequence[Chem.rdchem.Mol] = (),
        cutoff: float = 0.8,
        fp_type: str = "topological_torsion",
        similarity_metric: str = "dice",
    ) -> None:
        """Extract similar and matched structures

        Parameters
        ----------
        args : str
            Path to the .smi files
        queries : Sequence[Chem.rdchem.Mol], optional
            A list of molecule queries, by default an empty tuple
        patterns : Sequence[Chem.rdchem.Mol], optional
            A list of substructure patterns, by default an empty tuple
        cutoff : float, optional
            Similarity cutoff for extraction, by default 0.8
        fp_type : str, optional
            Fingerprint type for extraction, by default "topological_torsion"
        similarity_metric : str, optional
            Similarity metric for extraction, by default "dice"

        Returns
        -------
        None
        """

        # Read the .smi files
        df = self.read_smi(*args)

        queries = [query for query in queries if query is not None]
        if queries:
            queries_title = [query.GetProp("_Name") for query in queries]
            queries_fps = [gen_fp(query, fp_type) for query in queries]

            # Generate fingerprints and calculate similarities
            df[queries_title] = (
                df["mol"]
                .apply(gen_fp, fp_type=fp_type)
                .apply(calc_bulk_sim, args=(queries_fps, similarity_metric))
                .apply(pd.Series)
            )

            logging.info("Similarity calculation is successfully completed.")

            # Extract similar structures
            df["query_title"] = df[queries_title].idxmax(axis=1)
            df["similarity"] = df[queries_title].max(axis=1)
            df = df[df["similarity"] >= cutoff].reset_index(drop=True)

            # Drop the query columns
            df = df.drop(columns=queries_title)

        patterns = [pattern for pattern in patterns if pattern is not None]
        if patterns:
            patterns_smarts = [Chem.MolToSmarts(pattern) for pattern in patterns]

            # Substructure matching
            df[patterns_smarts] = (
                df["mol"].to_numpy() >= np.array(patterns).reshape(-1, 1)
            ).T

            logging.info("Substructure matching is successfully completed.")

            # Extract match structures
            df = df[df[patterns_smarts].any(axis=1)].reset_index(drop=True)
            df["pattern_smarts"] = df[patterns_smarts].idxmax(axis=1)

            # Drop the pattern columns
            df = df.drop(columns=patterns_smarts)

        # Generate molecular information
        df = self.gen_info(df)

        if df.empty:
            warn("No structures are extracted this time.")

        # Concatenate the dataframes (inner join)
        if self.structs.empty:
            self.structs = df
        else:
            self.structs = pd.concat([self.structs, df], ignore_index=True, join="inner")

        assert self.structs.notna().all().all()

        if self.structs.empty:
            warn("No structures are extracted yet.")

    def deduplicate(self) -> None:
        """Deduplicate the structures"""

        self.structs = self.structs.drop_duplicates(subset="smiles", ignore_index=True)

        logging.info("Deduplication is successfully completed.")

    def sort(self) -> None:
        """Sort the structures"""

        key = {
            "query_title": True,
            "similarity": False,
            "pattern_smarts": True,
            "cluster_size": False,
            "title": True,
        }

        by = [column for column in key if column in self.structs.columns]
        ascending = [key[column] for column in by]

        self.structs = self.structs.sort_values(
            by=by,
            ascending=ascending,
            ignore_index=True,
        )

        logging.info("Sorting is successfully completed.")

    def cluster(
        self,
        *,
        cutoff: float = 0.6,
        fp_type: str = "morgan",
        similarity_metric: str = "tanimoto",
    ) -> None:
        """Cluster the structures

        Parameters
        ----------
        cutoff : float, optional
            Similarity cutoff for clustering, by default 0.6
        fp_type : str, optional
            Fingerprint type for clustering, by default "morgan"
        similarity_metric : str, optional
            Similarity metric for clustering, by default "tanimoto"

        Returns
        -------
        None
        """

        # Reset the index to assert the correct mapping
        # self.structs = self.structs.reset_index(drop=True)
        assert np.all(self.structs.index == np.arange(len(self.structs)))

        fps = self.structs["mol"].apply(gen_fp, fp_type=fp_type).to_list()
        clusters = cluster_fps(fps, cutoff, similarity_metric)
        logging.info("Clustering is successfully completed.")

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

        assert np.all(df.index.sort_values() == np.arange(len(self.structs)))

        self.structs[columns] = df

    def write(
        self,
        output_dir: str,
        *,
        by: Literal["structs", "query", "pattern"] = "structs",
        xlsx: bool = False,
    ) -> None:
        """Write the output files

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        by : str, optional
            Group by 'structs', 'query', or 'pattern', by default "structs"
        xlsx : bool, optional
            Save as xlsx file with 2D structures, by default False

        Returns
        -------
        None
        """

        os.makedirs(output_dir, exist_ok=True)

        match by, xlsx:
            case "structs", True:
                PandasTools.SaveXlsxFromFrame(
                    self.structs,
                    os.path.join(output_dir, "structs.xlsx"),
                    molCol="mol",
                    size=(150, 150),
                )
                logging.info("Write structs.xlsx")
            case "structs", False:
                self.structs.drop(columns="mol").to_csv(
                    os.path.join(output_dir, "structs.csv"), index=False
                )
                logging.info("Write structs.csv")
            case "query", True:
                if "query_title" not in self.structs.columns:
                    raise ValueError("No query information is available.")
                for title, group in self.structs.groupby("query_title"):
                    PandasTools.SaveXlsxFromFrame(
                        group,
                        os.path.join(output_dir, f"{title}.xlsx"),
                        molCol="mol",
                        size=(150, 150),
                    )
                    logging.info("Write %s.xlsx", title)
            case "query", False:
                if "query_title" not in self.structs.columns:
                    raise ValueError("No query information is available.")
                for title, group in self.structs.groupby("query_title"):
                    group.drop(columns="mol").to_csv(
                        os.path.join(output_dir, f"{title}.csv"), index=False
                    )
                    logging.info("Write %s.csv", title)
            case "pattern", True:
                if "pattern_smarts" not in self.structs.columns:
                    raise ValueError("No pattern information is available.")
                for smarts, group in self.structs.groupby("pattern_smarts"):
                    PandasTools.SaveXlsxFromFrame(
                        group,
                        os.path.join(output_dir, f"{smarts}.xlsx"),
                        molCol="mol",
                        size=(150, 150),
                    )
                    logging.info("Write %s.xlsx", smarts)
            case "pattern", False:
                if "pattern_smarts" not in self.structs.columns:
                    raise ValueError("No pattern information is available.")
                for smarts, group in self.structs.groupby("pattern_smarts"):
                    group.drop(columns="mol").to_csv(
                        os.path.join(output_dir, f"{smarts}.csv"), index=False
                    )
                    logging.info("Write %s.csv", smarts)
            case _:
                raise ValueError(
                    "Invalid value for `by`, should be 'structs', 'query', or 'pattern'."
                )

    def draw(
        self,
        output_dir: str,
        *,
        by: Literal["structs", "query", "pattern"] = "structs",
        centroid: bool = False,
        **kwargs,
    ) -> None:
        """Draw 2D structures

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        by : str, optional
            Group by 'structs', 'query', or 'pattern', by default "structs"
        centroid : bool, optional
            Draw only centroids, by default False
        kwargs : Any
            Additional arguments for drawing 2D structures

        Returns
        -------
        None
        """

        os.makedirs(output_dir, exist_ok=True)

        match by, centroid:
            case "structs", True:
                if "cluster_centroid" not in self.structs.columns:
                    raise ValueError("No cluster information is available.")
                structs_centroid = self.structs[self.structs["cluster_centroid"]]
                draw_mols(
                    structs_centroid["mol"].tolist(),
                    os.path.join(output_dir, "centroids.png"),
                    legends=structs_centroid["title"].tolist(),
                    **kwargs,
                )
                logging.info("Draw centroids.png")
            case "structs", False:
                draw_mols(
                    self.structs["mol"].tolist(),
                    os.path.join(output_dir, "structs.png"),
                    legends=self.structs["title"].tolist(),
                    **kwargs,
                )
                logging.info("Draw structs.png")
            case "query", True:
                if "query_title" not in self.structs.columns:
                    raise ValueError("No query information is available.")
                if "cluster_centroid" not in self.structs.columns:
                    raise ValueError("No cluster information is available.")
                for title, group in self.structs.groupby("query_title"):
                    group_centroid = group[group["cluster_centroid"]]
                    legends = [
                        " ".join(x)
                        for x in zip(
                            group_centroid["title"],
                            group_centroid["similarity"].apply(lambda x: f"{x:.3f}"),
                        )
                    ]
                    draw_mols(
                        group_centroid["mol"].tolist(),
                        os.path.join(output_dir, f"{title}_centroids.png"),
                        legends=legends,
                        **kwargs,
                    )
                    logging.info("Draw %s_centroids.png", title)
            case "query", False:
                if "query_title" not in self.structs.columns:
                    raise ValueError("No query information is available.")
                for title, group in self.structs.groupby("query_title"):
                    legends = [
                        " ".join(x)
                        for x in zip(
                            group["title"],
                            group["similarity"].apply(lambda x: f"{x:.3f}"),
                        )
                    ]
                    draw_mols(
                        group["mol"].tolist(),
                        os.path.join(output_dir, f"{title}.png"),
                        legends=legends,
                        **kwargs,
                    )
                    logging.info("Draw %s.png", title)
            case "pattern", True:
                if "pattern_smarts" not in self.structs.columns:
                    raise ValueError("No pattern information is available.")
                if "cluster_centroid" not in self.structs.columns:
                    raise ValueError("No cluster information is available.")
                for smarts, group in self.structs.groupby("pattern_smarts"):
                    group_centroid = group[group["cluster_centroid"]]
                    draw_mols(
                        group_centroid["mol"].tolist(),
                        os.path.join(output_dir, f"{smarts}_centroids.png"),
                        legends=group_centroid["title"].tolist(),
                        pattern=Chem.MolFromSmarts(smarts),
                        **kwargs,
                    )
                    logging.info("Draw %s_centroids.png", smarts)
            case "pattern", False:
                if "pattern_smarts" not in self.structs.columns:
                    raise ValueError("No pattern information is available.")
                for smarts, group in self.structs.groupby("pattern_smarts"):
                    draw_mols(
                        group["mol"].tolist(),
                        os.path.join(output_dir, f"{smarts}.png"),
                        legends=group["title"].tolist(),
                        pattern=Chem.MolFromSmarts(smarts),
                        **kwargs,
                    )
                    logging.info("Draw %s.png", smarts)


def extract_smistructs(
    *args: str,
    output_dir: str | None = None,
    queries_file: str | None = None,
    patterns_smarts: Sequence[str] = (),
    extract_cutoff: float = 0.8,
    extract_fp_type: str = "topological_torsion",
    extract_similarity_metric: str = "dice",
    cluster_cutoff: float = 0.6,
    cluster_fp_type: str = "morgan",
    cluster_similarity_metric: str = "tanimoto",
    by: Literal["structs", "query", "pattern"] = "structs",
    xlsx: bool = False,
    centroid: bool = False,
    **kwargs: Any,
) -> None:
    """Extract similar and matched structures from .smi files

    Parameters
    ----------
    args : str
        Path to the .smi files
    output_dir : str, optional
        Path to the output directory, by default None,
        i.e., the same directory as the first input files
    queries_file : str, optional
        Path to the queries file, by default None
    patterns_smarts : Sequence[str], optional
        A list of substructure SMARTS patterns, by default an empty tuple
    extract_cutoff : float, optional
        Similarity cutoff for extraction, by default 0.8
    extract_fp_type : str, optional
        Fingerprint type for extraction, by default "topological_torsion"
    extract_similarity_metric : str, optional
        Similarity metric for extraction, by default "dice"
    cluster_cutoff : float, optional
        Similarity cutoff for clustering, by default 0.6
    cluster_fp_type : str, optional
        Fingerprint type for clustering, by default "morgan"
    cluster_similarity_metric : str, optional
        Similarity metric for clustering, by default "tanimoto"
    by : str, optional
        Group by 'structs', 'query', or 'pattern', by default "structs"
    xlsx : bool, optional
        Save as xlsx file with 2D structures, by default False
    centroid : bool, optional
        Draw only centroids, by default False
    kwargs : Any
        Additional arguments for drawing 2D structures

    Returns
    -------
    None
    """

    if queries_file is None:
        queries = []
    else:
        queries = [query for query in MolSupplier(queries_file) if query is not None]

    patterns = [
        pattern
        for smarts in patterns_smarts
        if (pattern := Chem.MolFromSmarts(smarts)) is not None
    ]

    smi_extractor = SmiExtractor()

    smi_extractor.extract(
        *args,
        queries=queries,
        patterns=patterns,
        cutoff=extract_cutoff,
        fp_type=extract_fp_type,
        similarity_metric=extract_similarity_metric,
    )

    smi_extractor.deduplicate()
    smi_extractor.sort()

    smi_extractor.cluster(
        cutoff=cluster_cutoff,
        fp_type=cluster_fp_type,
        similarity_metric=cluster_similarity_metric,
    )

    if output_dir is None:
        output_dir = os.path.dirname(args[0])
    smi_extractor.write(output_dir, by=by, xlsx=xlsx)
    smi_extractor.draw(output_dir, by=by, centroid=centroid, **kwargs)


def main() -> None:
    """Main function

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "smi_files",
        type=str,
        nargs="+",
        help="Path to the .smi files",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "-q",
        "--queries_file",
        type=str,
        help="Path to the queries file",
    )
    parser.add_argument(
        "-p",
        "--patterns_file",
        type=str,
        help="Path to the patterns file",
    )
    parser.add_argument(
        "-ec",
        "--extract_cutoff",
        type=float,
        default=0.8,
        help="Similarity cutoff for extraction",
    )
    parser.add_argument(
        "-ef",
        "--extract_fp_type",
        type=str,
        default="topological_torsion",
        help="Fingerprint type for extraction",
    )
    parser.add_argument(
        "-es",
        "--extract_similarity_metric",
        type=str,
        default="dice",
        help="Similarity metric for extraction",
    )
    parser.add_argument(
        "-cc",
        "--cluster_cutoff",
        type=float,
        default=0.6,
        help="Similarity cutoff for clustering",
    )
    parser.add_argument(
        "-cf",
        "--cluster_fp_type",
        type=str,
        default="morgan",
        help="Fingerprint type for clustering",
    )
    parser.add_argument(
        "-cs",
        "--cluster_similarity_metric",
        type=str,
        default="tanimoto",
        help="Similarity metric for clustering",
    )
    parser.add_argument(
        "-b",
        "--by",
        type=str,
        default="structs",
        choices=["structs", "query", "pattern"],
        help="Group by 'structs', 'query', or 'pattern'",
    )
    parser.add_argument(
        "-x",
        "--xlsx",
        action="store_true",
        help="Save as xlsx file with 2D structures",
    )
    parser.add_argument(
        "-c",
        "--centroid",
        action="store_true",
        help="Draw only centroids",
    )

    args = parser.parse_args()

    extract_smistructs(
        *args.smi_files,
        output_dir=args.output_dir,
        queries_file=args.queries_file,
        patterns_file=args.patterns_file,
        extract_cutoff=args.extract_cutoff,
        extract_fp_type=args.extract_fp_type,
        extract_similarity_metric=args.extract_similarity_metric,
        cluster_cutoff=args.cluster_cutoff,
        cluster_fp_type=args.cluster_fp_type,
        cluster_similarity_metric=args.cluster_similarity_metric,
        by=args.by,
        xlsx=args.xlsx,
        centroid=args.centroid,
    )


if __name__ == "__main__":
    main()

import os
import logging
import argparse
import multiprocessing as mp
from itertools import chain
from typing import Unpack, Sequence

import pandas as pd
from rdkit import Chem
from rdkit import DataStructs

# If you directly run this script, use `from tqdm import tqdm` instead of `... tqdm.notebook`
from tqdm.notebook import tqdm

from ._utils import (
    read_mols,
    is_pains,
    calc_descs,
    draw_structures,
    gen_fp,
    calc_sim,
)


def _compare_molecules(
    line: str,
    queries_title: Sequence[str],
    queries_fps: Sequence[DataStructs.cDataStructs.ExplicitBitVect],
    cutoff: float,
    fp_type: str,
    similarity_metric: str,
) -> list[list]:
    """Compare a molecule with a list of molecules

    Parameters
    ----------
    line : str
        A line of SMILES and title
    queries_title : Sequence[str]
        A list of query titles
    queries_fps : Sequence[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
        Fingerprints of molecules in queries
    cutoff : float
        Tanimoto similarity cutoff.
    fp_type : str
        Fingerprint type.
    similarity_metric : str
        Similarity metric.

    Returns
    -------
    similar_structures : list[list]
        A list of list of similar structures if any.
    """

    try:
        smiles, title = line.split()
    except ValueError:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    smiles = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True
    )  # Generate canonical SMILES

    similar_structures = []
    fp = gen_fp(mol, fp_type)
    for i, query_fp in enumerate(queries_fps):
        similarity = calc_sim(fp, query_fp, similarity_metric)
        if similarity > cutoff:
            canonical_smiles = Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=True
            )
            pains = is_pains(mol)
            descriptors = calc_descs(mol)
            similar_query = queries_title[i]
            similar_structures.append(
                [
                    title,
                    canonical_smiles,
                    pains,
                    *descriptors,  # type: ignore
                    similar_query,
                    similarity,
                ]
            )

    return similar_structures


def extract_similar_structures(
    queries_file: str,
    compounds_file: str,
    output_dir: str | None = None,
    cutoff: float = 0.8,
    *,
    fp_type: (
        str | Sequence[str]
    ) = "topological_torsion",  # TODO: Allow multiple fingerprints
    similarity_metric: (
        str | Sequence[str]
    ) = "dice",  # TODO: Allow multiple similarity metrics
    **kwargs,
) -> pd.DataFrame:
    """Extract similar structures from a compound library

    Parameters
    ----------
    queries_file : str
        Path to the query file.
    compounds_file : str
        Path to the compound file, should be a .smi file with a header.
    output_file : str, optional
        Path to the output file, by default None.
    cutoff : float, optional
        Tanimoto similarity cutoff, by default 0.8.
    fp_type : str | Sequence[str], optional
        Fingerprint type, by default `topological_torsion`.
    similarity_metric : str | Sequence[str], optional
        Similarity metric, by default `dice`.
    kwargs :
        Other keyword arguments for `draw_structures`.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe of similar structures.
    """

    # Read the query file
    queries = [mol for mol in read_mols(queries_file) if mol is not None]
    queries_title = [mol.GetProp("_Name") for mol in queries]
    queries_fps = [gen_fp(mol, fp_type) for mol in queries]  # type: ignore

    # Read the compound file
    if not compounds_file.endswith(".smi"):
        raise ValueError("The compound file should be a .smi file with a header")
    # Use `line`, `queries_title` and `queries_fps`
    # instead of `queries: list[rdkit.Chem.rdchem.Mol]` to save memory
    logging.info("Generate input arguments...")
    with open(compounds_file, "r", encoding="utf-8") as f:
        f.readline()  # Skip the header
        _star_args = [
            (
                line,
                queries_title,
                queries_fps,
                cutoff,
                fp_type,
                similarity_metric,
            )
            for line in f
        ]

    # Compare molecules in parallel
    logging.info("Start comparing molecules...")
    with mp.Pool(mp.cpu_count() - 1) as pool:
        raw_data = pool.starmap(
            _compare_molecules,
            tqdm(_star_args, total=len(_star_args), desc="Comparing", unit="mol"),
        )

    # Flatten the list
    data = list(chain.from_iterable(raw_data))

    # Write the output file
    logging.info("Start writing the output file...")
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
            "similar_query",
            "similarity",
        ],
    )
    df = df.sort_values(
        by=["similar_query", "similarity", "title"],
        ascending=[True, False, True],
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for query, group in df.groupby("similar_query"):
            group.to_csv(os.path.join(output_dir, f"{query}.csv"), index=False)
            logging.info("Write %s.csv", query)
            draw_structures(
                group["smiles"].apply(Chem.MolFromSmiles).tolist(),
                (
                    group["title"].tolist(),
                    group["similarity"].apply(lambda x: f"{x:.3f}").tolist(),
                ),
                os.path.join(output_dir, f"{query}.png"),
                **kwargs,
            )
            logging.info("Draw %s.png", query)

    logging.info("Completed!")

    return df


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
        required=True,
        help="Path to the query file",
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

    extract_similar_structures(
        args.queries,
        args.compounds,
        args.output,
        args.cutoff,
        fp_type=args.fp_type,
        similarity_metric=args.similarity_metric,
    )


if __name__ == "__main__":
    main()

import argparse
import logging
import multiprocessing as mp
import os
import re

import pandas as pd
from rdkit import Chem, DataStructs

from vspp._utils import calc_descs, cluster_fps, gen_fp, is_pains, read_mols, smart_tqdm


def _mmgbsa_dict(mmgbsa_txt: str) -> dict[str, float]:
    """Store the MM-GBSA data in a dictionary

    Parameters
    ----------
    mmgbsa_txt : str
        MM-GBSA txt file path.

    Returns
    -------
    mmgbsa_d : dict[str, float]
        A dictionary [title: MM-GBSA score].
    """

    if not os.path.exists(mmgbsa_txt):
        raise FileNotFoundError(f"{mmgbsa_txt} does not exist.")

    mmgbsa_d = {}  # Create a dictionary to store the MM-GBSA data
    with open(mmgbsa_txt, "r", encoding="utf-8") as mmgbsa_report:
        lines = mmgbsa_report.readlines()
        for line in reversed(
            lines
        ):  # Reverse the lines so that molecules with the lowest MM-GBSA scores are stored in the dictionary
            if match_line := re.match(
                r"\s*\d+\s+(\S+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", line
            ):
                title = match_line[1]
                mmgbsa_score = float(match_line[2])
                mmgbsa_d[title] = mmgbsa_score

    return mmgbsa_d


def _gen_mol_info(
    mol: Chem.rdchem.Mol,
    title: str,
    glide_score: float,
    mmgbsa_score: float,
    phase_score: float,
    fp_type: str,
) -> tuple[DataStructs.cDataStructs.ExplicitBitVect, list]:
    """Generate molecular information

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule
    title : str
        Molecule title
    glide_score : float
        Glide score
    mmgbsa_score : float
        MM-GBSA score
    phase_score : float
        Phase score
    fp_type : str
        Fingerprint type

    Returns
    -------
    fp : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        Morgan fingerprints
    phase_data : list
        A list of molecular information.
    """

    assert mol is not None, "Molecule is None."

    smiles = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True
    )  # Generate canonical isomeric SMILES
    pains = is_pains(mol)
    descriptors = calc_descs(mol)
    assert isinstance(descriptors, tuple)
    fp = gen_fp(mol, fp_type)
    phase_data = [
        title,
        smiles,
        glide_score,
        mmgbsa_score,
        phase_score,
        pains,
        *descriptors,
    ]
    return fp, phase_data


def analyze_mol_info(
    phase_maegz: str,
    mmgbsa_txt: str,
    output_file: str | None = None,
    cutoff: float = 0.6,
    fp_type: str = "morgan",
    similarity_metric: str = "tanimoto",
) -> pd.DataFrame:
    """Analyze molecular information from a phase maegz file and a MM-GBSA txt file

    Parameters
    ----------
    phase_maegz : str
        Phase maegz file path.
    mmgbsa_txt : str
        MM-GBSA txt file path.
    output_file : str, optional
        Output file path, by default None.
    cutoff : float, optional
        Tanimoto similarity cutoff, by default 0.6
    fp_type : str, optional
        Fingerprint type, by default "morgan"
    similarity_metric : str, optional
        Similarity metric, by default "tanimoto"

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing molecular information.
    """

    phase_suppl = read_mols(phase_maegz)
    mmgbsa_d = _mmgbsa_dict(mmgbsa_txt)

    _star_args = [
        (
            mol,
            mol.GetProp("_Name"),
            mol.GetProp("r_i_docking_score"),
            mmgbsa_d[mol.GetProp("_Name")],
            mol.GetProp("r_phase_PhaseScreenScore"),
            fp_type,
        )
        for mol in phase_suppl
    ]

    logging.info("Start generating molecular information...")

    # Generate molecular information in parallel
    with mp.Pool(mp.cpu_count() - 1) as pool:
        mol_info = pool.starmap(
            _gen_mol_info,
            smart_tqdm(_star_args, total=len(_star_args), desc="Analyzing", unit="mol"),
        )

    fps, phase_data = zip(*mol_info)

    logging.info("Start clustering molecules...")

    # Cluster the molecules by their Morgan fingerprints
    clusters = cluster_fps(
        fps, cutoff, similarity_metric
    )  # Cluster the molecules by their Morgan fingerprints
    for i, cluster in enumerate(clusters):
        for j, idx in enumerate(cluster):
            phase_data[idx].append(i + 1)  # Add the cluster ID
            phase_data[idx].append(len(cluster))  # Add the cluster size
            if j == 0:
                phase_data[idx].append(True)  # Add the cluster centroid
            else:
                phase_data[idx].append(False)

    logging.info("Start writing the output file...")

    # Write the output file
    df = pd.DataFrame(
        phase_data,
        columns=[
            "title",
            "smiles",
            "glide_score",
            "mmgbsa_score",
            "phase_score",
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
            "cluster_id",
            "cluster_size",
            "cluster_centroid",
        ],
    )
    df = df.sort_values(by=["cluster_id", "mmgbsa_score"])

    if output_file is not None:
        df.to_csv(output_file, index=False)

    return df


def main() -> None:
    """Main function

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(description="Postprocessing for virtual screening")
    parser.add_argument(
        "-p",
        "--phase_maegz",
        type=str,
        required=True,
        help="Phase maegz file path",
    )
    parser.add_argument(
        "-m",
        "--mmgbsa_txt",
        type=str,
        required=True,
        help="MM-GBSA txt file path",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=0.6,
        help="Tanimoto similarity cutoff",
    )
    parser.add_argument(
        "-f",
        "--fp_type",
        type=str,
        default="morgan",
        help="Fingerprint type",
    )
    parser.add_argument(
        "-s",
        "--similarity_metric",
        type=str,
        default="tanimoto",
        help="Similarity metric",
    )
    args = parser.parse_args()

    analyze_mol_info(
        args.phase_maegz,
        args.mmgbsa_txt,
        args.output_file,
        args.cutoff,
        args.fp_type,
        args.similarity_metric,
    )


if __name__ == "__main__":
    main()

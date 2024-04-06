import gzip
import os
from typing import Callable, Generator, Iterable, Sequence, Unpack
from warnings import warn

import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, FilterCatalog
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

DEFAULT_DESC_NAMES: set[str] = {name for name, _ in Descriptors.descList}

DESC_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "NumHAcceptors",
    "NumHDonors",
    "FractionCSP3",
    "NumRotatableBonds",
    "RingCount",
    "TPSA",
    "qed",
)

SIM_FUNCS: dict[str, Callable] = {
    name.lower(): func for name, func, _ in DataStructs.similarityFunctions
}


def calc_descs(
    mol: Chem.rdchem.Mol,
    desc_names: Sequence[str] | str = DESC_NAMES,
) -> tuple[float] | float:
    """Calculate molecular descriptors

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule
    desc_names : Sequence[str] | str, optional
        A list of descriptor names, by default `DESC_NAMES`, or a string of descriptor name.

    Returns
    -------
    descriptors : tuple[float] | float
        A list of molecular descriptors, or a single descriptor value
        `MolWt`, `MolLogP`, `NumHAcceptors` and `NumHDonors` are lipinski`s rule of five descriptors;
        `FractionCSP3`, `NumRotatableBonds`, `RingCount`, `TPSA` and `qed` are other common descriptors.
    """

    if isinstance(desc_names, str):
        if desc_names not in DEFAULT_DESC_NAMES:
            raise KeyError(f"Descriptor name should be in {DEFAULT_DESC_NAMES}.")
        # Directly pass a string to desc_names will cause every character to be a "descriptor name"
        # and return a tuple of interger 777 for UNKNOWN reasons
        calc = MoleculeDescriptors.MolecularDescriptorCalculator((desc_names,))
        return calc.CalcDescriptors(mol)[0]
    elif isinstance(desc_names, Sequence):
        if invalid_filter_keys := set(desc_names) - set(DEFAULT_DESC_NAMES):
            raise KeyError(f"Invalid descriptor names: {invalid_filter_keys}")
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
        return calc.CalcDescriptors(mol)
    else:
        assert False, "Descriptor names should be a string or a sequence of strings."


def calc_sim(
    fp1: DataStructs.cDataStructs.ExplicitBitVect,
    fp2: DataStructs.cDataStructs.ExplicitBitVect,
    similarity_metric: str = "tanimoto",
) -> float:
    """Calculate similarity between two fingerprints

    Parameters
    ----------
    fp1 : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        A fingerprint
    fp2 : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        A fingerprint
    similarity_metric : str, optional
        Similarity metric, by default 'tanimoto'

    Returns
    -------
    similarity : float
        Similarity between two fingerprints
    """

    if similarity_metric not in SIM_FUNCS:
        raise ValueError(
            "similarity_metric should be one of 'tanimoto', 'dice', 'cosine', 'sokal', 'russel', 'kulczynski', 'mcconnaughey', 'tversky' and 'asymmetric'."
        )

    return DataStructs.FingerprintSimilarity(
        fp1, fp2, metric=SIM_FUNCS[similarity_metric]
    )


def cluster_fps(
    fps: list[DataStructs.cDataStructs.ExplicitBitVect],
    cutoff: float = 0.6,
    similarity_metric: str = "tanimoto",
) -> list[tuple[int, ...]]:
    """Cluster the molecules by their Morgan fingerprints

    Parameters
    ----------
    fps : list[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
        A list of Morgan fingerprints
    cutoff : float, optional
        Tanimoto similarity cutoff, by default 0.6

    Returns
    -------
    clusters : list[tuple[int, ...]]
        A list of clusters sorted by cluster size
    """

    nfps = len(fps)
    # Calculate the distance matrix
    distances = [
        1 - calc_sim(fps[i], fps[j], similarity_metric=similarity_metric)
        for i in range(nfps)
        for j in range(i)
    ]

    # Cluster the molecules by their Morgan fingerprints
    clusters = list(Butina.ClusterData(distances, nfps, 1 - cutoff, isDistData=True))
    # Sort the clusters by the number of molecules in each cluster
    clusters.sort(key=len, reverse=True)

    return clusters


def draw_structures(
    mols: list[Chem.rdchem.Mol],
    output_file: str | None = None,
    *,
    legends: list[str] | None = None,
    pattern: Chem.rdchem.Mol | None = None,
    mols_per_row: int = 8,
    sub_img_size: tuple[float, float] = (300, 300),
) -> Image.Image | None:
    """Draw molecules

    Parameters
    ----------
    mols : list[rdkit.Chem.rdchem.Mol]
        A list of molecules
    output_file : str | None, optional
        Path to the output file, by default None
    legends : list[str] | None, optional
        A list of legends, by default None
    pattern : Chem.rdchem.Mol | None, optional
        SMARTS pattern to align and highlight the substructure, by default None
    mols_per_row : int, optional
        Number of molecules per row, by default 10
    sub_img_size : tuple[float, float], optional
        Size of each sub-image, by default (600, 600)

    Returns
    -------
    img : PIL.Image.Image | None
        A PIL image
    """

    Chem.rdDepictor.SetPreferCoordGen(True)

    max_mols = len(mols)

    if pattern is None:
        highlight_atom_lists = None
        highlight_bond_lists = None
    else:
        AllChem.Compute2DCoords(pattern)
        highlight_atom_lists = []
        highlight_bond_lists = []
        for mol in mols:
            if mol.HasSubstructMatch(pattern):
                # Align the molecule to the pattern
                AllChem.GenerateDepictionMatching2DStructure(mol, pattern)
                # Highlight the substructure
                highlight_atom = list(mol.GetSubstructMatch(pattern))
                highlight_atom_lists.append(highlight_atom)
                highlight_bond = [
                    mol.GetBondBetweenAtoms(
                        highlight_atom[bond.GetBeginAtomIdx()],
                        highlight_atom[bond.GetEndAtomIdx()],
                    ).GetIdx()
                    for bond in pattern.GetBonds()
                ]
                highlight_bond_lists.append(highlight_bond)
            else:
                highlight_atom_lists.append([])
                highlight_bond_lists.append([])

    try:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,  # molsPerRow should not be too small
            subImgSize=sub_img_size,
            legends=legends,
            highlightAtomLists=highlight_atom_lists,
            highlightBondLists=highlight_bond_lists,
            # returnPNG must be set EXPLICITLY to False
            # to avoid error in Jupyter Notebook
            returnPNG=False,
            maxMols=max_mols,  # maxMols must be set large enough to draw all molecules
        )
    except RuntimeError:
        warn(
            "`molsPerRow` is too small to draw a large number of molecules, try to increase it.",
            RuntimeWarning,
        )
        return None
    else:
        if output_file:
            img.save(output_file)

        return img


def filt_descs(
    mol: Chem.rdchem.Mol,
    filt: dict[str, tuple[float, float]],
) -> bool:
    """Filter the molecules based on the descriptors

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule
    filt : dict[str, tuple[float, float]]
        A dictionary of descriptor names and their ranges

    Returns
    -------
    filt_pass : bool
        True if the molecule passes the filter, False otherwise
    """

    if not filt:
        return True

    descs = np.array(calc_descs(mol, list(filt.keys())))  # type: ignore
    bounds = np.array(list(filt.values()))

    return bool(np.all((bounds[:, 0] <= descs) & (descs <= bounds[:, 1])))


def gen_fp(
    mol: Chem.rdchem.Mol,
    fp_type: str = "morgan",
) -> DataStructs.cDataStructs.ExplicitBitVect:
    """Generate molecular fingerprints

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule
    fp_type : str, optional
        Type of fingerprints, by default 'morgan'

    Returns
    -------
    fp : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        Fingerprints
    """

    fp_type = fp_type.lower()

    match fp_type:
        case "morgan":
            return AllChem.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
        case "ecfp2":
            return AllChem.GetMorganGenerator(radius=1, fpSize=1024).GetFingerprint(mol)
        case "ecfp4":  # ECFP4 is the same as Morgan fingerprints with radius=2
            return AllChem.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
        case "ecfp6":
            return AllChem.GetMorganGenerator(radius=3, fpSize=1024).GetFingerprint(mol)
        case "maccs":
            return Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        case "rdkit":
            return AllChem.GetRDKitFPGenerator().GetFingerprint(mol)
        case "atom_pair":
            return AllChem.GetAtomPairGenerator().GetFingerprint(mol)
        case "topological_torsion":
            return AllChem.GetTopologicalTorsionGenerator().GetFingerprint(mol)
        case _:
            raise ValueError(
                "fp_type should be one of 'morgan', 'ecfp2', 'ecfp4', 'ecfp6', "
                "'maccs', 'rdkit', 'atom_pair' and 'topological_torsion'."
            )


def is_pains(mol: Chem.rdchem.Mol) -> bool:
    """Identify and filiter out PAINS compounds

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule

    Returns
    -------
    pains : bool
        True if the molecule is a PAINS compound, False otherwise.
    """

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)
    return catalog.HasMatch(mol)


def read_mols(
    file: str, *, multithreaded=False
) -> Generator[Chem.rdchem.Mol, None, None]:
    """Read molecules from a file

    Parameters
    ----------
    file : str
        Path to the file, which can be a .sdf, .sdfgz, .mae, .maegz, .smi, .csv, .xlsx or .xls file.
    multithreaded : bool, optional
        Whether to use multithreading, by default False

    Returns
    -------
    mol_gen : Generator[rdkit.Chem.rdchem.Mol, None, None]
        A generator of molecules

    Notes
    -----
    The order of the molecules may not be preserved if `multithreaded` is True.
    In some cases, the multithreading may cause the program not to respond.
    """

    # Get the file extension
    ext = os.path.splitext(file)[1].lower()
    thread_num = 0
    queue_size = 1000
    match ext:
        case ".sdf":
            return (
                (
                    mol
                    for mol in Chem.MultithreadedSDMolSupplier(
                        file,
                        numWriterThreads=thread_num,
                        sizeInputQueue=queue_size,
                        sizeOutputQueue=queue_size,
                    )
                    if mol is not None
                )
                if multithreaded
                else (mol for mol in Chem.SDMolSupplier(file) if mol is not None)
            )
        case ".sdfgz":
            return (
                (
                    mol
                    for mol in Chem.MultithreadedSDMolSupplier(
                        gzip.open(file),  # type: ignore
                        numWriterThreads=thread_num,
                        sizeInputQueue=queue_size,
                        sizeOutputQueue=queue_size,
                    )
                    if mol is not None
                )
                if multithreaded
                else (
                    mol
                    for mol in Chem.SDMolSupplier(gzip.open(file))  # type: ignore
                    if mol is not None
                )
            )
        case ".mae":
            return (mol for mol in Chem.MaeMolSupplier(file) if mol is not None)
        case ".maegz":
            return (
                mol for mol in Chem.MaeMolSupplier(gzip.open(file)) if mol is not None
            )
        case ".smi":
            return (
                (
                    mol
                    for mol in Chem.MultithreadedSmilesMolSupplier(
                        file,
                        numWriterThreads=thread_num,
                        sizeInputQueue=queue_size,
                        sizeOutputQueue=queue_size,
                    )
                    if mol is not None
                )
                if multithreaded
                else (mol for mol in Chem.SmilesMolSupplier(file) if mol is not None)
            )
        case ".csv" | ".xlsx" | ".xls":
            df = pd.read_csv(file) if ext == ".csv" else pd.read_excel(file)

            def _process_df_row(row: pd.Series) -> Chem.rdchem.Mol:
                """Process a row of a dataframe to a molecule"""

                mol = Chem.MolFromSmiles(row["smiles"])
                mol.SetProp("_Name", row["title"])

                for prop, value in row.drop(["smiles", "title"]).items():
                    mol.SetProp(str(prop), str(value))

                return mol

            return (
                mol
                for _, row in df.iterrows()
                if (mol := _process_df_row(row)) is not None
            )
        case _:
            raise TypeError(
                "Should be a .sdf, .sdfgz, .mae, .maegz, .smi, .csv, .xlsx or .xls file."
            )


def smart_tqdm(iterable: Iterable, *args, **kwargs) -> tqdm | tqdm_notebook:
    """Smart tqdm

    Parameters
    ----------
    iterable : Iterable
        An iterable object
    args : tuple
        Positional arguments for tqdm
    kwargs : dict
        Keyword arguments for tqdm

    Returns
    -------
    tqdm : tqdm.tqdm | tqdm.notebook.tqdm
        A tqdm object
    """

    try:
        # Use get_ipython() in Jupyter notebook
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return tqdm_notebook(
                iterable, *args, **kwargs
            )  # Jupyter notebook or qtconsole
        raise RuntimeError  # Not in Jupyter, raise an error to trigger the fallback
    except (NameError, RuntimeError):
        return tqdm(iterable, *args, **kwargs)  # Probably standard Python interpreter

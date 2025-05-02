import gzip
import logging
import multiprocessing as mp
import os
from itertools import chain
from typing import Any, Callable, Generator, Iterable, Literal, Self, Sequence
from warnings import warn

import numpy as np
import pandas as pd
from numba import float32, int64, vectorize
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, FilterCatalog, PandasTools
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

type FingerPrint = DataStructs.cDataStructs.ExplicitBitVect

BUILTIN_DESC_NAMES: set[str] = {name for name, _ in Descriptors.descList}

COMMON_DESC_NAMES: tuple[str, ...] = (
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

SIM_FUNCS: dict[
    str,
    Callable[[FingerPrint, FingerPrint], float],
] = {name.lower(): func for name, func, _ in DataStructs.similarityFunctions}

BULK_SIM_FUNCS: dict[
    str,
    Callable[[FingerPrint, Sequence[FingerPrint]], list[float]],
] = {
    "tanimoto": DataStructs.cDataStructs.BulkTanimotoSimilarity,
    "dice": DataStructs.cDataStructs.BulkDiceSimilarity,
    "cosine": DataStructs.cDataStructs.BulkCosineSimilarity,
    "sokal": DataStructs.cDataStructs.BulkSokalSimilarity,
    "russel": DataStructs.cDataStructs.BulkRusselSimilarity,
    "rogotgoldberg": DataStructs.cDataStructs.BulkRogotGoldbergSimilarity,
    "allbit": DataStructs.cDataStructs.BulkAllBitSimilarity,
    "kulczynski": DataStructs.cDataStructs.BulkKulczynskiSimilarity,
    "mcconnaughey": DataStructs.cDataStructs.BulkMcConnaugheySimilarity,
    "asymmetric": DataStructs.cDataStructs.BulkAsymmetricSimilarity,
    "braunblanquet": DataStructs.cDataStructs.BulkBraunBlanquetSimilarity,
}


class MolSupplier:
    """Read and yield molecules from a file

    This class is a wrapper of RDKit's MolSupplier classes,
    which support multithreading and will skip empty molecules.

    Recommended for large files to avoid memory issues;
    please directly use rdkit.Chem.PandasTools for small files and .csv/.xlsx/.xls files.

    Attributes
    ----------
    file : str
        Path to the input file
    multithreaded : bool
        Whether to use multithreading
    thread_num : int
        Number of threads
    queue_size : int
        Size of the queue

    Methods
    -------
    __next__()
        Return the next molecule and skip empty molecules

    Raises
    ------
    TypeError
        If the file format is not supported
        or multithreading is not supported for the given file format

    Examples
    --------
    >>> supplier = MolSupplier("molecules.sdf")
    >>> for mol in supplier:
    ...     print(mol)

    Notes
    -----
    The order of the molecules is not guaranteed when using multithreading.
    """

    _THREAD_NUM: int = 0
    _QUEUE_SIZE: int = 1000

    def __init__(self, file: str, *, multithreaded=False, **kwargs) -> None:

        match os.path.splitext(file)[1].lower(), multithreaded:
            case ".sdf", False:
                self.mol_supplier = Chem.SDMolSupplier(file, **kwargs)
            case ".sdf", True:
                self.mol_supplier = Chem.MultithreadedSDMolSupplier(
                    file,
                    numWriterThreads=self._THREAD_NUM,
                    sizeInputQueue=self._QUEUE_SIZE,
                    sizeOutputQueue=self._QUEUE_SIZE,
                    **kwargs,
                )
            case ".sdfgz", False:
                self.mol_supplier = Chem.SDMolSupplier(gzip.open(file), **kwargs)
            case ".sdfgz", True:
                self.mol_supplier = Chem.MultithreadedSDMolSupplier(
                    gzip.open(file),
                    numWriterThreads=self._THREAD_NUM,
                    sizeInputQueue=self._QUEUE_SIZE,
                    sizeOutputQueue=self._QUEUE_SIZE,
                    **kwargs,
                )
            case ".mae", False:
                self.mol_supplier = Chem.MaeMolSupplier(file, **kwargs)
            case ".mae", True:
                raise TypeError("Multithreading is not supported for .mae files.")
            case ".maegz", False:
                self.mol_supplier = Chem.MaeMolSupplier(gzip.open(file), **kwargs)
            case ".maegz", True:
                raise TypeError("Multithreading is not supported for .maegz files.")
            case ".smi", False:
                self.mol_supplier = Chem.SmilesMolSupplier(
                    file, titleLine=False, **kwargs
                )
            case ".smi", True:
                self.mol_supplier = Chem.MultithreadedSmilesMolSupplier(
                    file,
                    titleLine=False,
                    numWriterThreads=self._THREAD_NUM,
                    sizeInputQueue=self._QUEUE_SIZE,
                    sizeOutputQueue=self._QUEUE_SIZE,
                    **kwargs,
                )
            case ".smr", False:
                with open(file, "r", encoding="utf-8") as f:
                    self.mol_supplier = (
                        Chem.MolFromSmarts(line, **kwargs) for line in f
                    )
            case _:
                raise TypeError("Should be a .sdf, .sdfgz, .mae, .maegz or .smi file.")

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Chem.rdchem.Mol:
        """Return the next molecule and skip empty molecules"""

        while True:
            if (mol := next(self.mol_supplier)) is not None:
                return mol
            warn("Empty molecule is skipped.", RuntimeWarning)


def calc_bulk_sim(
    fp: FingerPrint,
    fps: Sequence[FingerPrint],
    similarity_metric: str = "tanimoto",
) -> list[float]:
    """Calculate similarity between a fingerprint and a list of fingerprints

    Parameters
    ----------
    fp : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        A fingerprint
    fps : Sequence[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
        A list of fingerprints
    similarity_metric : str, optional
        Similarity metric, by default 'tanimoto'

    Returns
    -------
    similarities : list[float]
        Similarities between the fingerprint and a list of fingerprints
    """

    if similarity_metric not in BULK_SIM_FUNCS:
        raise ValueError(
            "similarity_metric should be one of 'tanimoto', 'dice', 'cosine', 'sokal', 'russel', 'rogotgoldberg', 'allbit', 'kulczynski', 'mcconnaughey', 'asymmetric', 'braunblanquet'."
        )

    return BULK_SIM_FUNCS[similarity_metric](fp, fps)


def calc_descs(
    mol: Chem.rdchem.Mol,
    *args: Sequence[str],
) -> float | tuple[float, ...]:
    """Calculate molecular descriptors

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule
    args : Sequence[str],
        Descriptor names, by default `DESC_NAMES`.

    Returns
    -------
    descriptors : float | tuple[float, ...]
        A single descriptor value or molecular descriptors.
        `MolWt`, `MolLogP`, `NumHAcceptors` and `NumHDonors` are `lipinski`s rule of five descriptors;
        `FractionCSP3`, `NumRotatableBonds`, `RingCount`, `TPSA` and `qed` are other common descriptors.
    """

    desc_names = args or COMMON_DESC_NAMES
    if invalid_filter_keys := set(desc_names) - set(BUILTIN_DESC_NAMES):
        raise KeyError(f"Invalid descriptor names: {invalid_filter_keys}")
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    # Directly pass a string to desc_names will cause every character to be a "descriptor name"
    # and return a tuple of interger 777 for UNKNOWN reasons
    return (
        calc.CalcDescriptors(mol)
        if len(desc_names) > 1
        else calc.CalcDescriptors(mol)[0]
    )


def calc_sim(
    fp1: FingerPrint,
    fp2: FingerPrint,
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
            "similarity_metric should be one of 'tanimoto', 'dice', 'cosine', 'sokal', 'russel', 'rogotgoldberg', 'allbit', 'kulczynski', 'mcconnaughey', 'asymmetric', 'braunblanquet'."
        )

    return DataStructs.FingerprintSimilarity(
        fp1, fp2, metric=SIM_FUNCS[similarity_metric]
    )


def cluster_fps(
    fps: Sequence[Any],
    cutoff: float = 0.6,
    similarity_metric: str = "tanimoto",
    *,
    multiprocessing: bool = False,
) -> list[tuple[int, ...]]:
    """Cluster the molecules by their fingerprints

    Parameters
    ----------
    fps : Sequence[Any]
        A list of fingerprints bit vectors
    cutoff : float, optional
        Similarity cutoff, by default 0.6
    similarity_metric : str, optional
        Similarity metric, by default 'tanimoto'
    multiprocessing : bool, optional
        Whether to use multiprocessing to calculate distance map, by default False
        `fps` should be a 2D numpy.ndarray with shape (n_mols, n_bits) if `multiprocessing` is True
        `multiprocessing` only supports 'tanimoto' and 'dice' similarity metrics

    Returns
    -------
    clusters : list[tuple[int, ...]]
        A list of clusters sorted by cluster size
    """

    nfps = len(fps)

    # Calculate the distance matrix
    logging.info("Calculating distances...")

    if multiprocessing:
        if not (isinstance(fps, np.ndarray) and fps.ndim == 2):
            raise ValueError(
                "`fps` should be a 2D numpy.ndarray with shape (n_mols, n_bits)."
            )
        if similarity_metric == "tanimoto":

            @vectorize([float32(int64)], target="parallel")
            def _calc_tanimoto_dist(n: int) -> float:
                i = int(np.sqrt(n + 1 / 8) * np.sqrt(2) + 1 / 2)
                j = n - i * (i - 1) // 2
                return 1 - np.sum(fps[i] & fps[j]) / np.sum(fps[i] | fps[j])

            distances = _calc_tanimoto_dist(np.arange(nfps * (nfps - 1) // 2))  # type: ignore
        elif similarity_metric == "dice":

            @vectorize([float32(int64)], target="parallel")
            def _calc_dice_dist(n: int) -> float:
                i = int(np.sqrt(n + 1 / 8) * np.sqrt(2) + 1 / 2)
                j = n - i * (i - 1) // 2
                return 1 - 2 * np.sum(fps[i] & fps[j]) / (
                    np.sum(fps[i]) + np.sum(fps[j])
                )

            distances = _calc_dice_dist(np.arange(nfps * (nfps - 1) // 2))  # type: ignore
        else:
            raise ValueError(
                "`similarity_metric` should be one of 'tanimoto' and 'dice'."
            )
    else:
        distances = np.empty(nfps * (nfps - 1) // 2)
        for i in tqdm(range(1, nfps)):
            distances[i * (i - 1) // 2 : i * (i + 1) // 2] = calc_bulk_sim(
                fps[i], fps[:i], similarity_metric
            )
        distances = 1 - distances

    # Cluster the molecules by their fingerprints
    logging.info("Clustering...")
    clusters = list(Butina.ClusterData(distances, nfps, 1 - cutoff, isDistData=True))
    logging.info("Molecules are clustered successfully!")

    # Sort the clusters by the number of molecules in each cluster
    clusters.sort(key=len, reverse=True)

    return clusters


def draw_mol(
    mol: Chem.rdchem.Mol,
    *,
    pattern: Chem.rdchem.Mol | None = None,
    img_size: tuple[float, float] = (300, 300),
    if_highlight_atoms: bool = True,
    alpha: float = 0.5,
) -> Image.Image | None:
    """Draw molecules

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule
    pattern : Chem.rdchem.Mol | None, optional
        SMARTS pattern to align and highlight the substructure, by default None
    img_size : tuple[float, float], optional
        Size of the image, by default (600, 600)
    if_highlight_atoms : bool, optional
        Whether to highlight the atoms, by default True
    alpha : float, optional
        Transparency of the highlight color, by default 0.5

    Returns
    -------
    img : PIL.Image.Image | None
        A PIL image
    """

    Chem.rdDepictor.SetPreferCoordGen(True)

    if pattern is None:
        highlight_atoms = []
        highlight_bonds = []
    else:
        AllChem.Compute2DCoords(pattern)
        if mol.HasSubstructMatch(pattern):
            # Align the molecule to the pattern
            AllChem.GenerateDepictionMatching2DStructure(mol, pattern)
            # Highlight the substructure
            highlight_atoms = list(mol.GetSubstructMatch(pattern))
            highlight_bonds = [
                mol.GetBondBetweenAtoms(
                    highlight_atoms[bond.GetBeginAtomIdx()],
                    highlight_atoms[bond.GetEndAtomIdx()],
                ).GetIdx()
                for bond in pattern.GetBonds()
            ]
        else:
            warn("The pattern is not found in the molecule.")

    return Draw.MolToImage(  # type: ignore
        mol,
        size=img_size,
        highlightAtoms=highlight_atoms if if_highlight_atoms else [],
        highlightBonds=highlight_bonds,
        highlightColor=(1, 0, 0, alpha),
    )


def draw_mols(
    mols: Sequence[Chem.rdchem.Mol],
    output_file: str | None = None,
    *,
    legends: Sequence[str] | None = None,
    pattern: Chem.rdchem.Mol | None = None,
    mols_per_row: int = 8,
    sub_img_size: tuple[float, float] = (300, 300),
    if_highlight_atoms: bool = True,
    # TODO: Add alpha to the function signature
    # alpha: float = 0.5,
) -> Image.Image | None:
    """Draw molecules

    Parameters
    ----------
    mols : Sequence[rdkit.Chem.rdchem.Mol]
        A list of molecules
    output_file : str | None, optional
        Path to the output file, by default None
    legends : Sequence[str] | None, optional
        A list of legends, by default None
    pattern : Chem.rdchem.Mol | None, optional
        SMARTS pattern to align and highlight the substructure, by default None
    mols_per_row : int, optional
        Number of molecules per row, by default 10
    sub_img_size : tuple[float, float], optional
        Size of each sub-image, by default (600, 600)
    if_highlight_atoms : bool, optional
        Whether to highlight the atoms, by default True
    alpha : float, optional
        Transparency of the highlight color, by default 0.5

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
            highlightAtomLists=highlight_atom_lists if if_highlight_atoms else None,
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
) -> bool | np.bool_:
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

    descs = np.array(calc_descs(mol, *filt.keys()))
    bounds = np.array(list(filt.values()))

    return np.all((bounds[:, 0] <= descs) & (descs <= bounds[:, 1]))


def gen_fp(
    mol: Chem.rdchem.Mol,
    fp_type: str = "morgan",
) -> FingerPrint:
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


def get_framework(
    mol: Chem.rdchem.Mol | str, *, generic: bool = False
) -> Chem.rdchem.Mol | str:
    """Get the Murcko scaffold of a molecule

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol | str
        A molecule or a SMILES string
    generic : bool, optional
        Whether to make the scaffold generic, by default False

    Returns
    -------
    scaffold : rdkit.Chem.rdchem.Mol | str
        Generic Murcko scaffold of the molecule
    """

    if isinstance(mol, str):
        if generic:
            return Chem.MolToSmiles(
                MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(mol))
                )
            )
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(mol)
    else:
        if generic:
            return MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)
            )
        return MurckoScaffold.GetScaffoldForMol(mol)


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

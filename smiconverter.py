import os
import logging
import argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import Self
from warnings import warn

from tqdm.notebook import tqdm
from rdkit import Chem

from ._utils import read_mols, filt_descs


class SMIConverter:
    """A class to convert a file to a .smi file

    This class is an iterator that yields SMILES and title of a molecule.

    Attributes
    ----------
    file : str
        Path to the input file
    prop : str, optional
        The property to be used as the title, by default "_Name"
    prefix : str, optional
        The prefix to be added to the title, by default ""
    filter : dict[str, tuple[float, float]] | None, optional
        The filter to be applied to the descriptors, by default None
    smi_ttl : list[tuple[str, str]]
        A list of tuples of SMILES and title
    num : int
        The number of molecules

    Methods
    -------
    deduplicate()
        Deduplicate the molecules
    sort()
        Sort the molecules
    write(output_dir: str, batch_size: int = 0)
        Write the .smi file (with optional batch size)
    """

    def __init__(
        self,
        file: str,
        *,
        prop: str = "_Name",
        prefix: str = "",
        filt: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the SMIConverter

        Parameters
        ----------
        file : str
            Path to the input file
        prop : str, optional
            The property to be used as the title, by default "_Name"
        prefix : str, optional
            The prefix to be added to the title, by default ""
        filt : dict[str, tuple[float, float]] | None, optional
            The filter to be applied to the descriptors, by default None

        Returns
        -------
        None
        """

        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} does not exist")
        self.file = file
        self.prop = prop
        self.prefix = prefix
        if filt is None:
            filt = {}
        self.filt: dict[str, tuple[float, float]] = filt

        logging.info("Converting %s to .smi file...", file)

        # _extract_smi_ttl is slower than multithreaded read_mols, but multiprocessing did not make it faster
        self.smi_ttl: list[tuple[str, str]] = [
            self._extract_smi_ttl(mol)
            for mol in tqdm(
                read_mols(self.file), desc="Reading", unit="mol"
            )  # Sometimes multithreaded read_mol will cause deadlock; please use single thread instead
            if filt_descs(mol, self.filt)
        ]
        self.num = len(self.smi_ttl)

        logging.info("%s molecules are successfully converted.", self.num)

    def __iter__(self) -> Self
        """Return the iterator object itself. This is required to be iterable."""

        return self

    def __next__(self) -> tuple[str, str]:
        """Get the next molecule in the file and convert it to SMILES and title"""

        if not self.smi_ttl:
            raise StopIteration
        return self.smi_ttl.pop(0)

    def __len__(self) -> int:
        """Return the number of molecules"""

        return self.num

    def __repr__(self) -> str:
        """Return the representation of the object"""

        return f"SMIConverter(file={self.file}, prop={self.prop}, prefix={self.prefix})"

    def _extract_smi_ttl(self, mol: Chem.rdchem.Mol) -> tuple[str, str] | None:
        """Convert a molecule to SMILES and title"""

        assert mol is not None, "Molecule is None."

        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if not mol.HasProp(self.prop):
            warn(f"{self.prop} does not exist in the molecule.")
            return smiles, "Unidentified"
        title = f"{self.prefix}{mol.GetProp(self.prop)}"
        return smiles, title

    def _write_batch(self, batch: list[tuple[str, str]], file_index: int) -> None:
        """Write a batch of molecules to a .smi file"""

        with open(
            f"{os.path.splitext(self.file)[0]}_{file_index}.smi",
            "w",
            encoding="utf-8",
        ) as f:
            f.write("smiles title\n")
            f.writelines(f"{smiles} {title}\n" for smiles, title in batch)

        logging.info(
            "Batch %d is written to %s_%d.smi",
            file_index,
            os.path.splitext(self.file)[0],
            file_index,
        )

    def deduplicate(self) -> None:
        """Deduplicate the molecules"""

        smiles_set = set()
        deduplicated = []
        for smiles, title in self.smi_ttl:
            if smiles not in smiles_set:
                smiles_set.add(smiles)
                deduplicated.append((smiles, title))

        self.smi_ttl = deduplicated
        self.num = len(self.smi_ttl)

        logging.info(
            "SMILES and title are deduplicated. %d molecules are left.", self.num
        )

    def sort(self) -> None:
        """Sort the molecules"""

        self.smi_ttl.sort(key=lambda x: x[1])

        logging.info("SMILES and title are sorted.")

    def write(self, output_dir: str, batch_size: int = 0) -> None:
        """Write the .smi file

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        batch_size : int, optional
            The batch size to split the .smi file, by default 0

        Returns
        -------
        None
        """

        if batch_size <= 0:
            self._write_batch(self.smi_ttl, 0)
            logging.info(
                "The .smi file is written to %s",
                os.path.join(output_dir, os.path.splitext(self.file)[0] + "_0.smi"),
            )
        else:
            with ThreadPoolExecutor(max_workers=mp.cpu_count() * 2 + 1) as executor:
                for i in range(0, self.num, batch_size):
                    executor.submit(
                        self._write_batch,
                        self.smi_ttl[i : i + batch_size],
                        i // batch_size,
                    )

            logging.info(
                "All .smi files are written to %s",
                os.path.join(output_dir, os.path.splitext(self.file)[0] + "_*.smi"),
            )


def convert_smi(
    input_file: str,
    *,
    prop: str = "_Name",
    prefix: str = "",
    filt: dict[str, tuple[float, float]] | None = None,
    deduplicate: bool = False,
    sort: bool = False,
    batch_size: int = 0,
):
    """Convert a file to a .smi file

    Parameters
    ----------
    input_file : str
        Path to the input file
    prop : str, optional
        The property to be used as the title, by default "_Name"
    prefix : str, optional
        The prefix to be added to the title, by default ""
    filt : dict[str, tuple[float, float]] | None, optional
        The filter to be applied to the descriptors, by default None
    deduplicate : bool, optional
        Deduplicate the molecules, by default False
    sort : bool, optional
        Sort the molecules, by default False
    batch_size : int, optional
        The batch size to split the .smi file, by default 0

    Returns
    -------
    None
    """

    smi_converter = SMIConverter(input_file, prop=prop, prefix=prefix, filt=filt)
    if deduplicate:
        smi_converter.deduplicate()
    if sort:
        smi_converter.sort()

    smi_converter.write(os.path.dirname(input_file), batch_size)


def main():
    """Main function

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(description="Convert a file to a .smi file")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument(
        "-p",
        "--prop",
        help="The property to be used as the title",
        default="_Name",
    )
    parser.add_argument(
        "-x",
        "--prefix",
        help="The prefix to be added to the title",
        default="",
    )
    parser.add_argument(
        "-f",
        "--filt",
        help="The filter to be applied to the descriptors",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--deduplicate",
        help="Deduplicate the molecules",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--sort",
        help="Sort the molecules",
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="The batch size to split the .smi file",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    convert_smi(
        args.input_file,
        prop=args.prop,
        prefix=args.prefix,
        filt=dict(
            zip(
                args.filt[::3],
                zip(map(float, args.filt[1::3]), map(float, args.filt[2::3])),
            )
        ),
        deduplicate=args.deduplicate,
        sort=args.sort,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

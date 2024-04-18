import argparse
import asyncio
import logging
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from typing import Sequence
from warnings import warn

import nest_asyncio
import pandas as pd
from rdkit import Chem

from vspp._utils import MolSupplier, filt_descs, smart_tqdm


class SmiConverter:
    """A class to convert files to .smi files"""

    def __init__(
        self,
        *,
        prop: str = "_Name",
        prefix: str = "",
        filt: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the SMIConverter

        Parameters
        ----------
        prop : str, optional
            The property to be used as the title, by default "_Name"
        prefix : str, optional
            The prefix to be added to the title, by default ""
        filt : dict[str, tuple[float, float]] | None, optional
            The filter to be applied to the descriptors, by default None,
            should be a dictionary of descriptor names and their ranges

        Returns
        -------
        None
        """

        self.prop = prop
        self.prefix = prefix
        if filt is None:
            filt = {}
        self.filt: dict[str, tuple[float, float]] = filt

        self.smi_ttl: list[tuple[str, str]] = []

    def _process_mols(self, mol: Chem.Mol) -> None:
        """Process a molecule"""

        assert mol is not None, "Molecule is None."

        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if mol.HasProp(self.prop):
            title = f"{self.prefix}{mol.GetProp(self.prop)}"
        else:
            warn(f"{self.prop} is not found in the molecule.")
            title = f"{self.prefix}Unidentified"

        self.smi_ttl.append((smiles, title))

    async def _async_generate_mols(
        self, file: str, q: asyncio.Queue, multithreaded: bool
    ) -> None:
        """Generate the molecules in the file asynchronously and put them in the queue"""

        for mol in smart_tqdm(
            MolSupplier(file, multithreaded=multithreaded),
            desc="Converting",
            unit="mol",
        ):
            if (mol is not None) and filt_descs(mol, self.filt):
                await q.put(mol)

        logging.info("All molecules are generated.")

    async def _async_process_mols(self, q: asyncio.Queue) -> None:
        """Process the molecules in the queue asynchronously"""

        while not q.empty():
            mol = await q.get()
            self._process_mols(mol)
            q.task_done()

        logging.info("All molecules are processed.")

    async def _async_convert(self, file: str, multithreaded: bool) -> None:
        """Convert the file to a .smi file"""

        q: asyncio.Queue = asyncio.Queue()

        producer = asyncio.create_task(
            self._async_generate_mols(file, q, multithreaded=multithreaded)
        )
        consumer = asyncio.create_task(self._async_process_mols(q))

        await asyncio.gather(producer, consumer)

    def convert(
        self, *args: str, asynchronous: bool = False, multithreaded: bool = False
    ) -> None:
        """Convert the file to a .smi file

        Parameters
        ----------
        args : str
            Path to the files to be converted
        asynchronous : bool, optional
            Whether to convert the file asynchronously, by default False
        multithreaded : bool, optional
            Whether to read the molecules in the file in a multithreaded way, by default False

        Returns
        -------
        None
        """

        if asynchronous:
            nest_asyncio.apply()
            for file in args:
                asyncio.run(self._async_convert(file, multithreaded=multithreaded))
        else:
            for file in args:
                for mol in smart_tqdm(
                    MolSupplier(file, multithreaded=multithreaded),
                    desc="Converting",
                    unit="mol",
                ):
                    if (mol is not None) and filt_descs(mol, self.filt):
                        self._process_mols(mol)

        logging.info("%s molecules are successfully converted.", len(self.smi_ttl))

    def deduplicate(self) -> None:
        """Deduplicate the molecules"""

        if not self.smi_ttl:
            raise ValueError("No molecules to deduplicate.")

        smiles_set = set()
        smiles_deduplicated = []
        for smiles, title in self.smi_ttl:
            if smiles not in smiles_set:
                smiles_set.add(smiles)
                smiles_deduplicated.append((smiles, title))

        self.smi_ttl = smiles_deduplicated

        logging.info(
            "SMILES and title are deduplicated. %d molecules are left.",
            len(self.smi_ttl),
        )

    def sort(self) -> None:
        """Sort the molecules"""

        if not self.smi_ttl:
            raise ValueError("No molecules to sort.")

        self.smi_ttl.sort(key=lambda x: x[1])

        logging.info("SMILES and title are sorted.")

    def _write_batch(
        self, chunk: Sequence[tuple[str, str]], output_dir: str, file_index: int
    ) -> None:
        """Write a batch of molecules to a .smi file"""

        if not chunk:
            return None

        with open(
            os.path.join(output_dir, f"lig_{file_index}.smi"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("smiles title\n")
            f.writelines(f"{smiles} {title}\n" for smiles, title in chunk)

        logging.info(
            "Chunk %d is written to %s_%d.smi",
            file_index,
            "lig",
            file_index,
        )

    def write(self, output_dir: str, chunk_size: int = 0) -> None:
        """Write the .smi file

        Parameters
        ----------
        output_dir : str
            Path to the output directory
        chunk_size : int, optional
            The chunk size to split the .smi file in the output, by default 0

        Returns
        -------
        None
        """

        if not self.smi_ttl:
            raise ValueError("No molecules to write.")

        os.makedirs(output_dir, exist_ok=True)

        if chunk_size <= 0:
            self._write_batch(self.smi_ttl, output_dir, 0)
            logging.info("The .smi file is written to %s", output_dir)
        else:
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                for i, chunk in enumerate(batched(self.smi_ttl, chunk_size)):
                    executor.submit(self._write_batch, chunk, output_dir, i)
            logging.info(
                "All .smi files are written to %s",
                output_dir,
            )


def files2smi(
    *args: str,
    output_dir: str | None = None,
    prop: str = "_Name",
    prefix: str = "",
    filt: dict[str, tuple[float, float]] | None = None,
    deduplicate: bool = False,
    sort: bool = False,
    batch_size: int = 0,
    asynchronous: bool = False,
    multithreaded: bool = False,
) -> None:
    """Convert a file to a .smi file

    Parameters
    ----------
    args : str
        Path to the files to be converted
    output_dir : str | None, optional
        Path to the output directory, by default None, 
        i.e., the same directory as the first input file
    prop : str, optional
        The property of molecules, especially in .sdf files, to be used as the title,
        by default "_Name", i.e., the default name of the molecule
    prefix : str, optional
        The prefix to be added to the title in .smi files for output,
        by default "", i.e., no prefix
    filt : dict[str, tuple[float, float]] | None, optional
        The filter to be applied to the descriptors, by default None,
        should be a dictionary of descriptor names and their ranges
    deduplicate : bool, optional
        Deduplicate the molecules, by default False
    sort : bool, optional
        Sort the molecules, by default False
    batch_size : int, optional
        The batch size to split the .smi file in the output, by default 0
    asynchronous : bool, optional
        Whether to convert the file asynchronously, by default False. Recommended for large files.
    multithreaded : bool, optional
        Whether to read the molecules in the file in a multithreaded way, by default False. Not recommended since it is not stable.

    Returns
    -------
    None
    """

    smi_converter = SmiConverter(prop=prop, prefix=prefix, filt=filt)

    smi_converter.convert(*args, asynchronous=asynchronous, multithreaded=multithreaded)

    if deduplicate:
        smi_converter.deduplicate()
    if sort:
        smi_converter.sort()

    if output_dir is None:
        output_dir = os.path.dirname(args[0])
    smi_converter.write(output_dir, batch_size)


def main():
    """Main function

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(description="Convert files to .smi files")
    parser.add_argument("input_files", nargs="+", help="Input files path")
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory path",
        default=None,
    )
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
        type=str,
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
        help="The batch size to split the .smi file, by default 0, i.e., no batches generated",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-a",
        "--asynchronous",
        help="Convert the file asynchronously. Recommended.",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--multithreaded",
        help="Read the molecules in the file in a multithreaded way. Not recommended.",
        action="store_true",
    )

    args = parser.parse_args()
    filt = dict(
        zip(
            args.filt[::3],
            zip(map(float, args.filt[1::3]), map(float, args.filt[2::3])),
        )
    )

    files2smi(
        *args.input_files,
        output_dir=args.output_dir,
        prop=args.prop,
        prefix=args.prefix,
        filt=filt,
        deduplicate=args.deduplicate,
        sort=args.sort,
        batch_size=args.batch_size,
        asynchronous=args.asynchronous,
        multithreaded=args.multithreaded,
    )


if __name__ == "__main__":
    main()

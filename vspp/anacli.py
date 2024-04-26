import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold

from vspp._pd_utils import cluster_df_frameworks, smi2df
from vspp._utils import draw_mol


class AnalogueClient:
    def __init__(self) -> None:
        self.library: pd.DataFrame | None = None
        self.frameworks: pd.DataFrame | None = None
        self.current_framework: str = ""
        self.selected_molecules: set[str] = set()

    def load_library(self, input_path: str) -> None:
        if input_path.endswith(".smi"):
            messagebox.showwarning(
                "Warning",
                "Loading a large library may take a while\nPlease be patient",
            )
            self.library = cluster_df_frameworks(smi2df(input_path)).set_index("title")
            self.frameworks = self.library.dropna().drop_duplicates(
                subset="cluster_framework"
            )  # Molecules without rings will be removed
            messagebox.showinfo(
                "Success",
                f"Library loaded successfully\n{len(self.library)} molecules found\n{len(self.frameworks)} frameworks",
            )
        elif input_path.endswith(".csv"):
            self.library = pd.read_csv(input_path)
            if {"smiles", "title", "cluster_framework"}.issubset(self.library.columns):
                self.library = self.library.set_index("title")
                self.frameworks = self.library.dropna().drop_duplicates(
                    subset="cluster_framework"
                )
                messagebox.showinfo(
                    "Success",
                    f"Library loaded successfully\n{len(self.library)} molecules found\n{len(self.frameworks)} frameworks",
                )
            else:
                messagebox.showerror("Error", "Invalid library format")
        else:
            messagebox.showerror("Error", "Invalid file format")

    def check_framework(self, framework_smiles: str) -> str:
        if (not framework_smiles) or (
            (framework := Chem.MolFromSmiles(framework_smiles)) is None
        ):
            messagebox.showerror("Error", "Invalid SMILES")
        else:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=framework)
        return ""

    def extract_molecules(self, framework_smiles: str) -> list[str]:
        if self.library is None:
            messagebox.showerror("Error", "Library not loaded")
        elif self.frameworks is None:
            messagebox.showerror("Error", "Frameworks not generated")
        elif framework_smiles:
            if framework_smiles in self.frameworks["cluster_framework"].values:
                self.current_framework = framework_smiles
                return self.library[
                    self.library["cluster_framework"] == framework_smiles
                ].index.tolist()
            else:
                messagebox.showerror("Error", "Framework not found")
        return []

    def title2smiles(self, title: str) -> str:
        if self.library is None:
            messagebox.showerror("Error", "Library not loaded")
        elif title:
            return str(self.library.loc[title, "smiles"])
        return ""

    def select_molecule(self, molecule_title: str):
        if self.library is None:
            messagebox.showerror("Error", "Library not loaded")
        elif molecule_title in self.library.index:
            self.selected_molecules.add(molecule_title)
        else:
            messagebox.showerror("Error", "Molecule not found")

    def remove_selection(self, molecule_title: str):
        if molecule_title in self.selected_molecules:
            self.selected_molecules.remove(molecule_title)
        else:
            messagebox.showerror("Error", "Molecule not selected")

    def save_analogues(self, output_path: str):
        if self.library is None:
            messagebox.showerror("Error", "Library not loaded")
        elif not self.selected_molecules:
            messagebox.showerror("Error", "No molecules selected")
        elif not output_path:
            messagebox.showerror("Error", "No output path provided")
        elif not output_path.endswith(".csv"):
            messagebox.showerror("Error", "Invalid output format")
        else:
            selected_molecules = self.library[
                self.library.index.isin(self.selected_molecules)
            ]
            selected_molecules.to_csv(output_path)
            messagebox.showinfo(
                "Success",
                f"Analogues saved successfully\n{len(selected_molecules)} molecules saved",
            )


class AnalogueGUI:
    def __init__(self, root: tk.Tk, client: AnalogueClient):
        self.root = root
        self.client = client
        self.root.title("Analogue Search")

        # Library file
        self.load_label = tk.Label(self.root, text="Input Library Path:")
        self.load_label.grid(row=0, column=0)
        self.load_entry = tk.Entry(self.root)
        self.load_entry.grid(row=0, column=1)
        self.library_browse_button = tk.Button(
            self.root, text="Browse", command=self.browse_input
        )
        self.library_browse_button.grid(row=0, column=2)
        self.library_load_button = tk.Button(
            self.root, text="Load", command=self.load_library
        )
        self.library_load_button.grid(row=0, column=3)

        # Framework SMILES
        self.framework_label = tk.Label(self.root, text="Framework SMILES:")
        self.framework_label.grid(row=1, column=0)
        self.framework_entry = tk.Entry(self.root)
        self.framework_entry.grid(row=1, column=1)
        self.framework_check_button = tk.Button(
            self.root, text="Check", command=self.check_framework
        )
        self.framework_check_button.grid(row=1, column=2)
        self.framework_clear_button = tk.Button(
            self.root, text="Clear", command=self.clear_framework
        )
        self.framework_clear_button.grid(row=1, column=3)

        # Molecule selection
        self.molecule_label = tk.Label(self.root, text="Select Molecule:")
        self.molecule_label.grid(row=1, column=4)
        self.molecule_combobox = ttk.Combobox(self.root, state="readonly")
        self.molecule_combobox.grid(row=1, column=5)
        self.molecule_combobox.bind("<<ComboboxSelected>>", self.display_molecule)
        self.molecule_select_button = tk.Button(
            self.root, text="Select", command=self.select_molecule
        )
        self.molecule_select_button.grid(row=1, column=6)

        # View selection
        self.view_label = tk.Label(self.root, text="View Selection:")
        self.view_label.grid(row=3, column=0)
        self.view_combobox = ttk.Combobox(self.root, state="readonly")
        self.view_combobox.grid(row=3, column=1)
        self.view_combobox.bind("<<ComboboxSelected>>", self.display_selection)
        self.molecule_deselect_button = tk.Button(
            self.root, text="Remove", command=self.remove_selection
        )
        self.molecule_deselect_button.grid(row=3, column=2)

        # Image areas
        self.framework_image_label = tk.Label(self.root, text="Framework")
        self.framework_image_label.grid(row=2, column=0, columnspan=3)
        self._clear_image(self.framework_image_label)
        self.molecule_image_label = tk.Label(self.root, text="Molecule")
        self.molecule_image_label.grid(row=2, column=4, columnspan=3)
        self._clear_image(self.molecule_image_label)
        self.selection_image_label = tk.Label(self.root, text="Selection")
        self.selection_image_label.grid(row=4, column=0, columnspan=3)
        self._clear_image(self.selection_image_label)

        # Save analogues
        self.save_label = tk.Label(self.root, text="Output Analogues Path:")
        self.save_label.grid(row=3, column=4)
        self.save_entry = tk.Entry(self.root)
        self.save_entry.grid(row=3, column=5)
        self.analogues_browse_button = tk.Button(
            self.root, text="Browse", command=self.browse_output
        )
        self.analogues_browse_button.grid(row=3, column=6)
        self.analogues_save_button = tk.Button(
            self.root, text="Save", command=self.save_analogues
        )
        self.analogues_save_button.grid(row=3, column=7)

    def _update_combobox(self, titles: list[str], combobox: ttk.Combobox):
        combobox["values"] = titles

    def _clear_combobox(self, combobox: ttk.Combobox, *, clear: bool = False):
        combobox.set("")
        if clear:
            combobox["values"] = []

    def _update_image(self, smiles, label, *, pattern: Chem.rdchem.Mol | None = None):
        try:
            mol = Chem.MolFromSmiles(smiles)
            img = draw_mol(mol, pattern=pattern)
        except ValueError:
            messagebox.showerror("Error", "Invalid SMILES")
        except RuntimeError:
            messagebox.showerror("Error", "Unable to generate image")
        else:
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo)
            label.image = photo

    def _clear_image(self, label):
        self._update_image("", label)

    def browse_input(self):
        input_path = filedialog.askopenfilename(
            defaultextension=".smi",
            initialdir="./",
            title="Browse Input",
            filetypes=[
                ("SMI Files", "*.smi"),
                ("CSV Files", "*.csv"),
                ("All Files", "*.*"),
            ],
        )
        self.load_entry.delete(0, tk.END)
        self.load_entry.insert(0, input_path)
        self.load_library()

    def load_library(self):
        input_path = self.load_entry.get()
        self.client.load_library(input_path)

    def check_framework(self):
        framework_smiles = self.framework_entry.get()
        framework_smiles = self.client.check_framework(framework_smiles)
        self._update_image(framework_smiles, self.framework_image_label)
        titles = self.client.extract_molecules(framework_smiles)
        self._update_combobox(titles, self.molecule_combobox)
        self._clear_image(self.molecule_image_label)
        self._clear_combobox(self.molecule_combobox)

    def clear_framework(self):
        self.framework_entry.delete(0, tk.END)
        self._clear_image(self.framework_image_label)
        self._clear_image(self.molecule_image_label)
        self._clear_combobox(self.molecule_combobox, clear=True)

    def display_molecule(self, event: tk.Event):
        molecule_title = self.molecule_combobox.get()
        molecule_smiles = self.client.title2smiles(molecule_title)
        self._update_image(
            molecule_smiles,
            self.molecule_image_label,
            pattern=Chem.MolFromSmiles(self.client.current_framework),
        )

    def display_selection(self, event: tk.Event):
        selected_title = self.view_combobox.get()
        selected_smiles = self.client.title2smiles(selected_title)
        self._update_image(selected_smiles, self.selection_image_label)

    def select_molecule(self):
        selected_molecule = self.molecule_combobox.get()
        self.client.select_molecule(selected_molecule)
        self._update_combobox(list(self.client.selected_molecules), self.view_combobox)

    def remove_selection(self):
        removed_selection = self.view_combobox.get()
        self.client.remove_selection(removed_selection)
        self._update_combobox(list(self.client.selected_molecules), self.view_combobox)
        self._clear_image(self.selection_image_label)
        self._clear_combobox(self.view_combobox)

    def browse_output(self):
        output_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir="./",
            title="Save Analogues",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        self.save_entry.delete(0, tk.END)
        self.save_entry.insert(0, output_path)
        self.save_analogues()

    def save_analogues(self):
        output_path = self.save_entry.get()
        self.client.save_analogues(output_path)


def main():
    root = tk.Tk()
    app = AnalogueGUI(root, AnalogueClient())
    root.mainloop()


if __name__ == "__main__":
    main()

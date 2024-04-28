import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from vspp._pd_utils import cluster_df_frameworks, gen_df_info, smi2df
from vspp._utils import draw_mol
from vspp.smiextractor import extract_query


class AnalogueClient:
    def __init__(self) -> None:
        self.library: pd.DataFrame = pd.DataFrame()
        self.frameworks: pd.DataFrame = pd.DataFrame()
        self.selected_molecules: set[str] = set()

    def load_library(self, input_path: str) -> None:
        if not input_path:
            pass
        elif input_path.endswith(".smi"):
            self.library = cluster_df_frameworks(smi2df(input_path)).set_index("title")
            self.frameworks = self.library.dropna().drop_duplicates(
                subset="cluster_framework"
            )  # Molecules without rings will be removed
            self.library.to_csv(f"{os.path.splitext(input_path)[0]}_clustered.csv")
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
                messagebox.showerror("Error", "Invalid file format")
        else:
            messagebox.showerror("Error", "Invalid file format")

    def check_smiles(self, smiles: str) -> str:
        if self.library.empty:
            messagebox.showerror("Error", "Library not loaded")
        elif not smiles:
            messagebox.showerror("Error", "No SMILES provided")
        elif (query := Chem.MolFromSmiles(smiles)) is None:
            messagebox.showerror("Error", "Invalid SMILES")
        else:
            return Chem.MolToSmiles(query)
        return ""

    def fetch_framework(self, smiles: str) -> str:
        if self.frameworks.empty:
            messagebox.showerror("Error", "Frameworks not generated")
        elif smiles:
            framework_smiles = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
            if framework_smiles not in self.frameworks["cluster_framework"].values:
                messagebox.showwarning("Warning", "Framework not found in the library")
            return framework_smiles  # type: ignore
        else:
            assert False
        return ""

    def search_frameworks(self, framework_smiles: str) -> list[str]:
        if self.frameworks.empty:
            messagebox.showerror("Error", "Frameworks not generated")
        elif not framework_smiles:
            messagebox.showerror("Error", "No SMILES provided")
        elif frameworks := extract_query(
            self.frameworks,
            Chem.MolFromSmiles(framework_smiles),
            smi_col="cluster_framework",
            cutoff=0.6,
        )["cluster_framework"].to_list():
            messagebox.showinfo(
                "Success",
                f"{len(frameworks)} framework(s) found",
            )
            return frameworks
        else:
            messagebox.showerror("Error", "No frameworks found")
        return []

    def extract_molecules(self, framework_smiles: str) -> list[str]:
        if self.library.empty:
            messagebox.showerror("Error", "Library not loaded")
        elif self.frameworks.empty:
            messagebox.showerror("Error", "Frameworks not generated")
        elif not framework_smiles:
            messagebox.showerror("Error", "No SMILES provided")
        elif framework_smiles in self.frameworks["cluster_framework"].values:
            return self.library[
                self.library["cluster_framework"] == framework_smiles
            ].index.to_list()
        else:
            messagebox.showerror("Error", "Framework not found")
        return []

    def select_molecule(self, molecule_title: str) -> None:
        if self.library.empty:
            messagebox.showerror("Error", "Library not loaded")
        elif molecule_title in self.library.index:
            self.selected_molecules.add(molecule_title)
        else:
            messagebox.showerror("Error", "Molecule not found")

    def remove_selection(self, molecule_title: str) -> None:
        if not self.selected_molecules:
            messagebox.showerror("Error", "No molecules selected")
        elif molecule_title in self.selected_molecules:
            self.selected_molecules.remove(molecule_title)
        else:
            messagebox.showerror("Error", "Molecule not selected")

    def save_analogues(self, output_path: str) -> None:
        if self.library.empty:
            messagebox.showerror("Error", "Library not loaded")
        elif not self.selected_molecules:
            messagebox.showerror("Error", "No molecules selected")
        elif not output_path:
            pass
        elif not output_path.endswith(".csv"):
            messagebox.showerror("Error", "Invalid file format")
        else:
            selected_molecules = self.library[
                self.library.index.isin(self.selected_molecules)
            ]
            selected_molecules = gen_df_info(selected_molecules)
            selected_molecules.to_csv(output_path)
            messagebox.showinfo(
                "Success",
                f"Analogues saved successfully\n{len(selected_molecules)} molecules saved",
            )


class AnalogueGUI:
    def __init__(self, root: tk.Tk, client: AnalogueClient):
        # TODO: Add `Copy` and `Next` buttons

        self.root: tk.Tk = root
        self.client: AnalogueClient = client
        self.current_framework: str = ""
        self.root.title("Analogue Search")

        # Load library
        self.load_library_label = tk.Label(self.root, text="Input Library Path:")
        self.load_library_label.grid(row=0, column=0)
        self.load_library_entry = tk.Entry(self.root)
        self.load_library_entry.grid(row=0, column=1)
        self.load_library_browse_button = tk.Button(
            self.root, text="Browse", command=self.browse_input
        )
        self.load_library_browse_button.grid(row=0, column=2)
        self.load_library_load_button = tk.Button(
            self.root, text="Load", command=self.load_library
        )
        self.load_library_load_button.grid(row=0, column=3)

        # Parse SMILES
        self.parse_smiles_label = tk.Label(self.root, text="Parse SMILES:")
        self.parse_smiles_label.grid(row=1, column=0)
        self.parse_smiles_entry = tk.Entry(self.root)
        self.parse_smiles_entry.grid(row=1, column=1)
        self.parse_smiles_check_button = tk.Button(
            self.root, text="Check", command=self.check_smiles
        )
        self.parse_smiles_check_button.grid(row=1, column=2)
        self.parse_smiles_clear_button = tk.Button(
            self.root, text="Clear", command=self.clear_all
        )
        self.parse_smiles_clear_button.grid(row=1, column=3)
        self.parse_smiles_image_label = tk.Label(self.root)
        self.parse_smiles_image_label.grid(row=2, column=0, columnspan=3)
        self._update_image("", self.parse_smiles_image_label)

        # Retrieve frameworks
        self.retrieve_frameworks_label = tk.Label(
            self.root, text="Retrieve Frameworks:"
        )
        self.retrieve_frameworks_label.grid(row=1, column=4)
        self.retrieve_frameworks_combobox = ttk.Combobox(self.root, state="readonly")
        self.retrieve_frameworks_combobox.grid(row=1, column=5)
        self.retrieve_frameworks_combobox.bind(
            "<<ComboboxSelected>>", self._display_framework
        )
        self.retrieve_frameworks_search_button = tk.Button(
            self.root, text="Search", command=self.search_frameworks
        )
        self.retrieve_frameworks_search_button.grid(row=1, column=6)
        self.retrieve_frameworks_submit_button = tk.Button(
            self.root, text="Submit", command=self.submit_framework
        )
        self.retrieve_frameworks_submit_button.grid(row=1, column=7)
        self.retrieve_frameworks_image_label = tk.Label(self.root)
        self.retrieve_frameworks_image_label.grid(row=2, column=4, columnspan=3)
        self._update_image("", self.retrieve_frameworks_image_label)

        # Choose molecule
        self.choose_molecule_label = tk.Label(self.root, text="Choose Molecule:")
        self.choose_molecule_label.grid(row=3, column=0)
        self.choose_molecule_combobox = ttk.Combobox(self.root, state="readonly")
        self.choose_molecule_combobox.grid(row=3, column=1)
        self.choose_molecule_combobox.bind(
            "<<ComboboxSelected>>", self._display_molecule
        )
        self.choose_molecule_confirm_button = tk.Button(
            self.root, text="Confirm", command=self.confirm_molecule
        )
        self.choose_molecule_confirm_button.grid(row=3, column=2)
        self.choose_molecule_image_label = tk.Label(self.root)
        self.choose_molecule_image_label.grid(row=4, column=0, columnspan=3)
        self._update_image("", self.choose_molecule_image_label)

        # Review selections
        self.review_selections_label = tk.Label(self.root, text="Review Selections:")
        self.review_selections_label.grid(row=3, column=4)
        self.review_selections_combobox = ttk.Combobox(self.root, state="readonly")
        self.review_selections_combobox.grid(row=3, column=5)
        self.review_selections_combobox.bind(
            "<<ComboboxSelected>>", self._display_selection
        )
        self.review_selections_remove_button = tk.Button(
            self.root, text="Remove", command=self.remove_selection
        )
        self.review_selections_remove_button.grid(row=3, column=6)
        self.review_selections_image_label = tk.Label(self.root)
        self.review_selections_image_label.grid(row=4, column=4, columnspan=3)
        self._update_image("", self.review_selections_image_label)

        # Save analogues
        self.save_analogues_label = tk.Label(self.root, text="Output Analogues Path:")
        self.save_analogues_label.grid(row=0, column=4)
        self.save_analogues_entry = tk.Entry(self.root)
        self.save_analogues_entry.grid(row=0, column=5)
        self.save_analogues_browse_button = tk.Button(
            self.root, text="Browse", command=self.browse_output
        )
        self.save_analogues_browse_button.grid(row=0, column=6)
        self.save_analogues_save_button = tk.Button(
            self.root, text="Save", command=self.save_analogues
        )
        self.save_analogues_save_button.grid(row=0, column=7)

    def _clear_parse_smiles(self):
        self.parse_smiles_entry.delete(0, tk.END)
        self._update_image("", self.parse_smiles_image_label)

    def _clear_retrieve_frameworks(self):
        self.retrieve_frameworks_combobox.set("")
        self.retrieve_frameworks_combobox["values"] = []
        self.current_framework = ""
        self._update_image("", self.retrieve_frameworks_image_label)

    def _clear_choose_molecule(self):
        self.choose_molecule_combobox.set("")
        self.choose_molecule_combobox["values"] = []
        self._update_image("", self.choose_molecule_image_label)

    def _clear_review_selections(self):
        self.review_selections_combobox.set("")
        self.review_selections_combobox["values"] = []
        self._update_image("", self.review_selections_image_label)

    def _display_framework(self, event: tk.Event | None = None):
        framework_smiles = self.retrieve_frameworks_combobox.get()
        self._update_image(framework_smiles, self.retrieve_frameworks_image_label)

    def _display_molecule(self, event: tk.Event | None = None):
        molecule_title = self.choose_molecule_combobox.get()
        molecule_smiles = str(self.client.library.loc[molecule_title, "smiles"])
        self._update_image(
            molecule_smiles,
            self.choose_molecule_image_label,
            pattern_smiles=self.current_framework,
        )

    def _display_selection(self, event: tk.Event | None = None):
        selected_title = self.review_selections_combobox.get()
        selected_smiles = str(self.client.library.loc[selected_title, "smiles"])
        self._update_image(selected_smiles, self.review_selections_image_label)

    def _update_image(self, smiles: str, label: tk.Label, *, pattern_smiles: str = ""):
        try:
            mol = Chem.MolFromSmiles(smiles)
            pattern = Chem.MolFromSmiles(pattern_smiles) if pattern_smiles else None
            img = draw_mol(mol, pattern=pattern, alpha=0.25, if_highlight_atoms=False)
        except ValueError:
            messagebox.showerror("Error", "Invalid SMILES")
        except RuntimeError:
            messagebox.showerror("Error", "Unable to generate image")
        else:
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo)
            label.image = photo

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
        self.load_library_entry.delete(0, tk.END)
        self.load_library_entry.insert(0, input_path)
        self.load_library()

    def load_library(self):
        input_path = self.load_library_entry.get()
        self.client.load_library(input_path)
        self.clear_all()

    def check_smiles(self):
        smiles = self.parse_smiles_entry.get()
        smiles = self.client.check_smiles(smiles)
        self._clear_choose_molecule()
        self._clear_retrieve_frameworks()
        self._update_image(smiles, self.parse_smiles_image_label)
        if framework_smiles := self.client.fetch_framework(smiles):
            self.retrieve_frameworks_combobox["values"] = [framework_smiles]
            self.retrieve_frameworks_combobox.set(framework_smiles)
            self._display_framework()

    def clear_all(self):
        self._clear_choose_molecule()
        self._clear_retrieve_frameworks()
        self._clear_parse_smiles()

    def search_frameworks(self):
        framework_smiles = self.retrieve_frameworks_combobox.get()
        frameworks = self.client.search_frameworks(framework_smiles)
        self._clear_choose_molecule()
        self._clear_retrieve_frameworks()
        self.retrieve_frameworks_combobox["values"] = frameworks

    def submit_framework(self):
        self.current_framework = self.retrieve_frameworks_combobox.get()
        molecules = self.client.extract_molecules(self.current_framework)
        self._clear_choose_molecule()
        self.choose_molecule_combobox["values"] = molecules
        if molecules:
            self.choose_molecule_combobox.set(molecules[0])
            self._display_molecule()

    def confirm_molecule(self):
        selected_molecule = self.choose_molecule_combobox.get()
        self.client.select_molecule(selected_molecule)
        self.review_selections_combobox["values"] = list(self.client.selected_molecules)
        self.review_selections_combobox.set(selected_molecule)
        self._display_selection()

    def remove_selection(self):
        removed_selection = self.review_selections_combobox.get()
        self.client.remove_selection(removed_selection)
        self._clear_review_selections()
        self.review_selections_combobox["values"] = list(self.client.selected_molecules)

    def browse_output(self):
        output_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir="./",
            title="Save Analogues",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        self.save_analogues_entry.delete(0, tk.END)
        self.save_analogues_entry.insert(0, output_path)
        self.save_analogues()

    def save_analogues(self):
        output_path = self.save_analogues_entry.get()
        self.client.save_analogues(output_path)


def main():
    root = tk.Tk()
    app = AnalogueGUI(root, AnalogueClient())
    root.mainloop()


if __name__ == "__main__":
    main()

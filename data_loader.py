"""data_loader.py

Loads and indexes all three input files at startup.  Every other module
imports from here; no other module touches the raw files directly.

Design decisions
----------------
- Prevalence is stored as a fraction [0, 1] (xlsx values are %, divided by 100).
- background_prevalence[hpo_id] = mean prevalence across all 17 modules for
  that term.  Used as fallback when the term is absent from a module profile.
- GLOBAL_FLOOR = 0.001 (0.1%) is used when a term is absent from every module.
- Gene HPO profiles are built as an inverted index from the per-module sheets:
  for each (hpo_id, gene_list) row, every gene listed gets that hpo_id added
  to its profile.  Genes belong to exactly one module.
- The `ird_terms` property exposes all HPO IDs that appear in any module sheet;
  used by hpo_traversal to filter get_children results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Minimum probability floor used when a term is completely absent from profiles.
GLOBAL_FLOOR: float = 0.001

# Column names in the xlsx that we actually use.
_COL_HPO_ID = "hpo_id"
_COL_HPO_NAME = "phenotype_name"
_COL_PREVALENCE = "target_module_phenotype_prevalence_percent"
_COL_GENES = "target_module_genes_with_phenotype"

NUM_MODULES = 17
MODULE_IDS = list(range(NUM_MODULES))


class DataLoader:
    """Singleton-style loader; instantiate once and pass around."""

    def __init__(self, data_dir: str = "Input") -> None:
        self._data_dir = Path(data_dir)
        self._prevalence: dict[tuple[str, int], float] = {}   # (hpo_id, module_id) -> [0,1]
        self._background: dict[str, float] = {}               # hpo_id -> mean prevalence
        self._hpo_name: dict[str, str] = {}                   # hpo_id -> term name
        self._name_to_id: dict[str, str] = {}                 # lower-case name -> hpo_id
        self._module_genes: dict[int, list[str]] = {}         # module_id -> [gene, ...]
        self._gene_info: dict[str, dict] = {}                 # gene -> {module_id, stability_score, classification}
        self._gene_hpo: dict[str, set[str]] = {}              # gene -> {hpo_id, ...}
        self._signatures: dict[int, list[dict]] = {}          # module_id -> [sig_info, ...]
        self._signature_terms: dict[str, set[int]] = {}       # hpo_id -> {module_id, ...}
        self._ird_terms: frozenset[str] = frozenset()

        module_dfs = self._load_xlsx()
        self._load_gene_csv()
        self._load_signatures()
        # gene_hpo index requires both xlsx and gene_info to be loaded
        self._build_gene_hpo_index(module_dfs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prevalence(self, hpo_id: str, module_id: int) -> float:
        """Return prevalence of hpo_id in module as a fraction [0, 1].

        Falls back to background mean, then to GLOBAL_FLOOR.
        """
        key = (hpo_id, module_id)
        if key in self._prevalence:
            return self._prevalence[key]
        return self._background.get(hpo_id, GLOBAL_FLOOR)

    def get_background(self, hpo_id: str) -> float:
        """Return mean prevalence across all 17 modules (background rate)."""
        return self._background.get(hpo_id, GLOBAL_FLOOR)

    @property
    def ird_terms(self) -> frozenset[str]:
        """All HPO IDs present in at least one module sheet (annotation space)."""
        return self._ird_terms

    @property
    def hpo_name(self) -> dict[str, str]:
        """Map hpo_id -> human-readable term name."""
        return self._hpo_name

    @property
    def name_to_id(self) -> dict[str, str]:
        """Map lower-case term name -> hpo_id."""
        return self._name_to_id

    @property
    def module_genes(self) -> dict[int, list[str]]:
        """Map module_id -> list of gene symbols."""
        return self._module_genes

    @property
    def gene_info(self) -> dict[str, dict]:
        """Map gene -> {module_id, stability_score, classification}."""
        return self._gene_info

    @property
    def gene_hpo(self) -> dict[str, set[str]]:
        """Map gene -> set of HPO IDs annotated to that gene (within its module)."""
        return self._gene_hpo

    @property
    def signatures(self) -> dict[int, list[dict]]:
        """Map module_id -> list of significant phenotypic signatures."""
        return self._signatures

    @property
    def signature_terms(self) -> dict[str, set[int]]:
        """Map hpo_id -> set of module_ids for which it is a significant signature."""
        return self._signature_terms

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_xlsx(self) -> dict[int, pd.DataFrame]:
        # Updated to April 13 all-HPO file
        path = self._data_dir / "module_all_HPO_background_comparison_20260413_1019.xlsx"
        xl = pd.ExcelFile(path)

        # Accumulate per-term prevalences across modules to compute background.
        prevalence_by_term: dict[str, list[float]] = {}
        module_dfs: dict[int, pd.DataFrame] = {}

        for module_id in MODULE_IDS:
            sheet_name = f"module_{module_id}"
            df = xl.parse(sheet_name)
            module_dfs[module_id] = df

            for _, row in df.iterrows():
                hpo_id: str = row[_COL_HPO_ID]
                name: str = row[_COL_HPO_NAME]
                pct: float = float(row[_COL_PREVALENCE])

                # Convert percent -> fraction
                prev = pct / 100.0

                self._prevalence[(hpo_id, module_id)] = prev

                # Name lookups (values are identical across sheets)
                if hpo_id not in self._hpo_name:
                    self._hpo_name[hpo_id] = name
                    self._name_to_id[name.lower()] = hpo_id

                prevalence_by_term.setdefault(hpo_id, []).append(prev)

        # Background = mean across all 17 modules for each term
        self._background = {
            hpo_id: sum(vals) / len(vals)
            for hpo_id, vals in prevalence_by_term.items()
        }
        self._ird_terms = frozenset(self._background.keys())
        return module_dfs

    def _load_gene_csv(self) -> None:
        # P3-A: Updated to newest April 12th version
        path = self._data_dir / "gene_classification_20260412_1524.csv"
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            gene: str = row["gene"]
            module_id: int = int(row["module_id"])
            self._gene_info[gene] = {
                "module_id": module_id,
                "stability_score": float(row["stability_score"]),
                "classification": str(row["classification"]),
            }
            self._module_genes.setdefault(module_id, []).append(gene)

    def _load_signatures(self) -> None:
        # P3-A: Updated to newest April 12th version
        path = self._data_dir / "module_phenotypic_signatures_FDR_corrected_20260412_1524.csv"
        if not path.exists():
            return
            
        df = pd.read_csv(path)
        # Filter for significant signatures only
        df = df[df["fdr_significant"] == True]

        for _, row in df.iterrows():
            m_id = int(row["module_id"])
            h_id = str(row["hpo_term"])
            
            sig_info = {
                "hpo_id": h_id,
                "term_name": self._hpo_name.get(h_id, h_id),
                "odds_ratio": float(row["odds_ratio"]),
                "q_value": float(row["q_value"]),
                "freq_in_module": float(row["freq_in_module"]),
                "specificity_ratio": float(row["specificity_ratio"])
            }
            
            self._signatures.setdefault(m_id, []).append(sig_info)
            self._signature_terms.setdefault(h_id, set()).add(m_id)

    def _build_gene_hpo_index(self, module_dfs: dict[int, pd.DataFrame]) -> None:
        """Invert the per-module gene lists to get per-gene HPO profiles.

        Only rows with prevalence > 0 are indexed; zero-prevalence rows mean
        no gene in the module has that phenotype.
        """
        # Updated to April 13 all-HPO file
        known_genes: set[str] = set(self._gene_info.keys())

        for module_id in MODULE_IDS:
            df = module_dfs[module_id]
            df = df[df[_COL_PREVALENCE] > 0]

            for _, row in df.iterrows():
                hpo_id: str = row[_COL_HPO_ID]
                raw_genes = row[_COL_GENES]

                if not isinstance(raw_genes, str) or not raw_genes.strip():
                    continue

                for gene in (g.strip() for g in raw_genes.split(",")):
                    if gene and gene in known_genes:
                        self._gene_hpo.setdefault(gene, set()).add(hpo_id)

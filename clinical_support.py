"""clinical_support.py

Public API entry point.  No scoring logic lives here — this module only
orchestrates calls to data_loader, hpo_traversal, scoring_engine,
gene_ranker, prediction_engine, and session_manager.

Design decisions
----------------
- The engine is initialized lazily at first call to avoid slow startup
  in import-only contexts (e.g., Streamlit hot-reload).  Use
  ClinicalSupportEngine(eager=True) to force initialization upfront.

- `query_gene(gene_symbol)` finds the gene's assigned module, scores
  using only the gene's annotated HPO terms as observed, and returns a
  QueryResult focused on that module.

- `_build_result()` is a module-level function (not a method) so that
  Session.get_current_result() can call it without importing the engine
  class, avoiding a circular import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from output_models import QueryResult, ModuleMatch
from data_loader import DataLoader, MODULE_IDS
from hpo_traversal import HPOTraversal
from scoring_engine import ScoringEngine
from gene_ranker import GeneRanker
from prediction_engine import PredictionEngine
from session_manager import Session
from ethnicity_prior_policy import EthnicityPriorPolicy

logger = logging.getLogger(__name__)

# Default data directory relative to this file
_DEFAULT_DATA_DIR = Path(__file__).parent / "Input"
_EBL_DIR = Path(__file__).parent / "ethnicity_bayes_layer"


class ClinicalSupportEngine:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        eager: bool = False,
        gamma: float = 0.3,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._dl: DataLoader | None = None
        self._ht: HPOTraversal | None = None
        self._se: ScoringEngine | None = None
        self._gr: GeneRanker | None = None
        self._pe: PredictionEngine | None = None
        self._gamma = gamma
        self._ebl_lr: pd.DataFrame | None = None
        self._ebl_cnt: pd.DataFrame | None = None
        self._ebl_policy: EthnicityPriorPolicy | None = None

        if eager:
            self._init()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ethnicity_group(self) -> str:
        return ""

    @property
    def use_ethnicity_prior(self) -> bool:
        return False

    @property
    def ebl_lr_matrix(self) -> pd.DataFrame | None:
        self._init()
        return self._ebl_lr

    @property
    def ebl_count_matrix(self) -> pd.DataFrame | None:
        self._init()
        return self._ebl_cnt

    @property
    def ebl_ethnicity_totals(self) -> dict[str, int]:
        self._init()
        return getattr(self, "_ebl_eth_totals", {})

    def _init(self) -> None:
        """Initialize all sub-modules (idempotent)."""
        if self._dl is not None:
            return
        self._dl = DataLoader(str(self._data_dir))
        self._ht = HPOTraversal(self._data_dir / "hp.obo", self._dl.ird_terms)
        self._se = ScoringEngine(self._dl, self._ht)

        # Load EBL matrices (fail silently — layer stays off if files absent)
        try:
            self._ebl_lr = pd.read_csv(_EBL_DIR / "lr_matrix_all.csv", index_col=0)
            self._ebl_cnt = pd.read_csv(_EBL_DIR / "count_matrix_all.csv", index_col=0)
            ts = pd.read_csv(_EBL_DIR / "training_set.csv")
            self._ebl_eth_totals = ts["eth_group"].value_counts().to_dict()
        except FileNotFoundError:
            logger.warning("EBL matrices not found at %s — ethnicity prior disabled.", _EBL_DIR)
            self._ebl_lr = None
            self._ebl_cnt = None
            self._ebl_eth_totals = {}

        self._ebl_policy = EthnicityPriorPolicy.default()

        self._gr = GeneRanker(
            self._dl, self._ht,
            gamma=self._gamma,
            lr_matrix=self._ebl_lr,
            count_matrix=self._ebl_cnt,
            ebl_policy=self._ebl_policy,
        )
        self._pe = PredictionEngine(self._dl, self._ht, self._se)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        observed: list[str] | None = None,
        excluded: list[str] | None = None,
        ethnicity_group: str | None = None,
        use_ethnicity_prior: bool | None = None,
    ) -> QueryResult:
        """Full phenotype query.

        Parameters
        ----------
        observed : list of HPO IDs present in the patient
        excluded : list of HPO IDs confirmed absent
        ethnicity_group : per-query ethnicity context (optional)
        use_ethnicity_prior : whether to apply EBL for this query (optional)
        """
        self._init()
        effective_ethnicity = ethnicity_group or ""
        effective_use_prior = bool(use_ethnicity_prior) if use_ethnicity_prior is not None else False
        if effective_use_prior and effective_ethnicity:
            logger.info(
                "Applied ethnicity prior [%s] to phenotype query "
                "(%d observed, %d excluded terms).",
                effective_ethnicity,
                len(observed or []),
                len(excluded or []),
            )
        return _build_result(
            self._dl, self._se, self._gr, self._pe,
            observed or [], excluded or [],
            ethnicity_group=effective_ethnicity,
            use_ethnicity_prior=effective_use_prior,
        )

    def query_gene(
        self,
        gene_symbol: str,
        ethnicity_group: str | None = None,
        use_ethnicity_prior: bool | None = None,
    ) -> QueryResult:
        """Gene-first query.

        Uses the gene's annotated HPO terms as 'observed' to produce a
        result focused on the gene's assigned module.
        """
        self._init()
        gene = gene_symbol.strip().upper()
        info = self._dl.gene_info.get(gene)
        if info is None:
            raise ValueError(f"Gene not found in IRD panel: {gene!r}")
        effective_ethnicity = ethnicity_group or ""
        effective_use_prior = bool(use_ethnicity_prior) if use_ethnicity_prior is not None else False
        if effective_use_prior and effective_ethnicity:
            logger.info(
                "Applied ethnicity prior [%s] to gene-first query for %s.",
                effective_ethnicity,
                gene,
            )
        gene_hpos = sorted(self._dl.gene_hpo.get(gene, set()))
        return _build_result(
            self._dl, self._se, self._gr, self._pe,
            gene_hpos, [],
            ethnicity_group=effective_ethnicity,
            use_ethnicity_prior=effective_use_prior,
        )

    def gene_observed_hpo_ids(self, gene_symbol: str) -> list[str]:
        """HPO IDs used as the observed set in query_gene (sorted), for UI metrics."""
        self._init()
        gene = gene_symbol.strip().upper()
        return sorted(self._dl.gene_hpo.get(gene, set()))

    def new_session(
        self,
        ethnicity_group: str | None = None,
        use_ethnicity_prior: bool | None = None,
    ) -> Session:
        """Create a new interactive Q&A session."""
        self._init()
        effective_ethnicity = ethnicity_group or ""
        effective_use_prior = bool(use_ethnicity_prior) if use_ethnicity_prior is not None else False
        return Session(
            self._dl,
            self._se,
            self._gr,
            self._pe,
            ethnicity_group=effective_ethnicity,
            use_ethnicity_prior=effective_use_prior,
        )

    # ------------------------------------------------------------------
    # UI support — data retrieval for display (no clinical logic)
    # ------------------------------------------------------------------

    def get_hpo_options(self) -> list[tuple[str, str]]:
        """Return (hpo_id, term_name) pairs sorted by prevalence descending for autocomplete.

        Includes only IRD-annotated terms with max prevalence >= 1% in any
        module (~3 000 terms), which is a manageable size for st.multiselect.
        """
        self._init()
        opts: list[tuple[float, str, str]] = []
        for hpo_id in self._dl.ird_terms:
            max_prev = max(self._dl.get_prevalence(hpo_id, m) for m in MODULE_IDS)
            if max_prev >= 0.01:
                name = self._dl.hpo_name.get(hpo_id, hpo_id)
                opts.append((max_prev, hpo_id, name))
        opts.sort(key=lambda x: x[0], reverse=True)
        return [(x[1], x[2]) for x in opts]

    def get_gene_options(self) -> list[str]:
        """Return sorted list of all 442 IRD gene symbols."""
        self._init()
        return sorted(self._dl.gene_info.keys())

    def get_term_name(self, hpo_id: str) -> str:
        """Return the human-readable term name for an HPO ID."""
        self._init()
        return self._dl.hpo_name.get(hpo_id, hpo_id)

    def is_module_signature(self, hpo_id: str, module_id: int) -> bool:
        """Return True if hpo_id is a significant signature for module_id."""
        self._init()
        return module_id in self._dl.signature_terms.get(hpo_id, set())

    def browse_module(self, module_id: int) -> dict:
        """Return structured display data for the Module Browser view.

        Returns a plain dict (no dataclasses) so app.py can build DataFrames
        directly without any scoring or clinical logic.
        """
        self._init()

        # Top HPO terms, excluding near-universal generic taxonomy nodes
        terms: list[dict] = []
        for hpo_id in self._dl.ird_terms:
            prev = self._dl.get_prevalence(hpo_id, module_id)
            if prev > 0 and self._dl.get_background(hpo_id) < 0.80:
                terms.append({
                    "HPO ID": hpo_id,
                    "Term": self._dl.hpo_name.get(hpo_id, hpo_id),
                    "Module prevalence": f"{prev * 100:.1f}%",
                    "_sort": prev,
                })
        terms.sort(key=lambda x: x["_sort"], reverse=True)
        for t in terms:
            del t["_sort"]

        # Module-specific signatures from the new file
        sigs: list[dict] = []
        for s in self._dl.signatures.get(module_id, []):
            sigs.append({
                "HPO ID": s["hpo_id"],
                "Term": s["term_name"],
                "Odds Ratio": round(s["odds_ratio"], 2),
                "q-value": f"{s['q_value']:.2e}",
                "Freq in Module": f"{s['freq_in_module'] * 100:.1f}%",
                "Specificity Ratio": round(s["specificity_ratio"], 2)
            })
        sigs.sort(key=lambda x: float(x["Odds Ratio"]) if x["Odds Ratio"] != float('inf') else 1e9, reverse=True)

        # Genes with all relevant display fields
        gene_rows: list[dict] = []
        for g in self._gr.rank_genes(module_id, []):
            info = self._dl.gene_info.get(g.gene, {})
            gene_rows.append({
                "Gene": g.gene,
                "Stability": g.stability.capitalize(),
                "Stability score": round(info.get("stability_score", 0.0), 4),
                "HPO annotations": len(self._dl.gene_hpo.get(g.gene, set())),
            })

        return {
            "module_id": module_id,
            "gene_count": len(self._dl.module_genes.get(module_id, [])),
            "annotated_term_count": len(terms),
            "top_terms": terms,
            "signatures": sigs,
            "signature_count": len(sigs),
            "genes": gene_rows,
        }


# ------------------------------------------------------------------
# Module-level helper (used by both ClinicalSupportEngine and Session)
# ------------------------------------------------------------------

def _build_result(
    dl: DataLoader,
    se: ScoringEngine,
    gr: GeneRanker,
    pe: PredictionEngine,
    observed: list[str],
    excluded: list[str],
    ethnicity_group: str = "",
    use_ethnicity_prior: bool = False,
) -> QueryResult:
    """Assemble a full QueryResult from raw observed/excluded lists."""
    posterior = se.score_modules(observed, excluded)
    confidence = se.compute_confidence(posterior)

    top_module_id = max(posterior, key=posterior.get)

    all_modules = sorted(
        [
            ModuleMatch(
                module_id=m,
                module_label=None,  # reserved for manual clinical annotation
                probability=round(posterior[m], 6),
                supporting_phenotypes=_supporting_for_module(dl, observed, m),
            )
            for m in MODULE_IDS
        ],
        key=lambda mm: mm.probability,
        reverse=True,
    )

    top_module = next(mm for mm in all_modules if mm.module_id == top_module_id)

    candidate_genes = gr.rank_genes(
        top_module_id,
        observed,
        ethnicity_group=ethnicity_group,
        use_ethnicity_prior=use_ethnicity_prior,
    )
    phenotype_predictions = pe.predict_phenotypes(top_module_id, observed)
    next_questions = pe.suggest_next_questions(posterior, observed, excluded, k=5)
    if not next_questions:
        next_question = pe.suggest_next_question(posterior, observed, excluded)
        next_questions = [next_question]
    else:
        next_question = next_questions[0]

    return QueryResult(
        top_module=top_module,
        all_modules=all_modules,
        candidate_genes=candidate_genes,
        phenotype_predictions=phenotype_predictions,
        next_question=next_question,
        confidence=round(confidence, 6),
        next_questions=next_questions,
    )


def _supporting_for_module(
    dl: DataLoader,
    observed: list[str],
    module_id: int,
) -> list[str]:
    """Return observed HPO term names with prevalence > 0 in the given module."""
    return [
        dl.hpo_name.get(h, h)
        for h in observed
        if dl.get_prevalence(h, module_id) > 0
    ]

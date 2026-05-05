"""discovery_manager.py

Track 2: Discovery & Suggestion Panel.

EvidenceSource is a structural duck-type protocol — any object with a
`name` attribute and a `get_candidates()` method qualifies.
DiscoveryManager aggregates results from all registered live sources
and flags genes supported by more than one source as Master Candidates.

Live sources
------------
EBLSource   Ethnicity Bayes Layer (lr_matrix_all.csv + count_matrix_all.csv)

Planned sources (UI stubs, no logic yet)
-----------------------------------------
NPP         Network Phylogenetic Profiling — protein co-evolution scores
PPI         Protein–Protein Interaction network proximity
"""
from __future__ import annotations

import pandas as pd

from output_models import DiscoverySuggestion
from ethnicity_prior_policy import EthnicityPriorPolicy

# Metadata cards for sources that are planned but not yet implemented
PLANNED_SOURCES: list[dict] = [
    {
        "name": "NPP",
        "label": "Network Phylogenetic Profiling",
        "description": (
            "Gene co-evolution scores across phylogenetic profiles. "
            "Genes with high NPP similarity to known IRD genes in this module "
            "will be surfaced as candidates."
        ),
    },
    {
        "name": "PPI",
        "label": "Protein–Protein Interaction",
        "description": (
            "Network proximity in the human interactome. "
            "Genes within 1–2 interaction hops of confirmed module members "
            "will be ranked by connectivity."
        ),
    },
]


class EBLSource:
    """Ethnicity Bayes Layer evidence source.

    Returns genes that (1) are absent from the primary engine gene set,
    (2) have Ethnicity LR >= min_lr, and (3) have training count >= min_n
    in the patient's ethnic group.
    """

    name = "EBL"

    def __init__(
        self,
        lr_matrix: pd.DataFrame,
        count_matrix: pd.DataFrame,
        policy: EthnicityPriorPolicy | None = None,
    ) -> None:
        self._lr = lr_matrix
        self._cnt = count_matrix
        self._policy = policy or EthnicityPriorPolicy.default()
        self._min_lr = self._policy.min_lr_upweight
        self._min_n = self._policy.min_n_upweight

    def get_candidates(
        self,
        excluded_genes: set[str],
        context: dict,
    ) -> dict[str, dict]:
        """Return {gene: metadata} for genes that pass the Expert Gate.

        Parameters
        ----------
        excluded_genes : the engine's full gene set (442 IRD genes).
                         Any gene in this set is already in the primary
                         channel and must not appear in Track 2.
        context        : must contain key "ethnicity_group" (str).
        """
        eth = str(context.get("ethnicity_group", "")).strip()
        if not eth or eth not in self._lr.columns:
            return {}

        results: dict[str, dict] = {}
        for gene in self._lr.index:
            if gene in excluded_genes:
                continue
            lr_val = self._lr.at[gene, eth]
            if pd.isna(lr_val) or float(lr_val) < self._min_lr:
                continue
            n = 0
            if gene in self._cnt.index and eth in self._cnt.columns:
                n = int(self._cnt.at[gene, eth])
            if n < self._min_n:
                continue
            results[gene] = {
                "lr": round(float(lr_val), 3),
                "n": n,
                "ethnicity": eth,
            }
        return results


class DiscoveryManager:
    """Aggregates evidence from all registered live sources.

    Parameters
    ----------
    sources : list of EvidenceSource-compatible objects
    """

    def __init__(self, sources: list) -> None:
        self._sources = sources

    def get_suggestions(
        self,
        excluded_genes: set[str],
        context: dict,
    ) -> list[DiscoverySuggestion]:
        """Return discovery candidates sorted by evidence strength.

        Genes supported by >1 source are flagged as Master Candidates
        and sorted to the top.

        Parameters
        ----------
        excluded_genes : full engine gene set — genes to exclude from Track 2
        context        : e.g. {"ethnicity_group": "North_African_Jewish"}
        """
        per_source: dict[str, dict[str, dict]] = {}
        for src in self._sources:
            per_source[src.name] = src.get_candidates(excluded_genes, context)

        all_genes: set[str] = set()
        for candidates in per_source.values():
            all_genes.update(candidates.keys())

        suggestions: list[DiscoverySuggestion] = []
        for gene in all_genes:
            active_sources = [
                name for name, cands in per_source.items() if gene in cands
            ]
            metadata: dict[str, dict] = {
                src: per_source[src][gene] for src in active_sources
            }
            suggestions.append(
                DiscoverySuggestion(
                    gene=gene,
                    sources=active_sources,
                    source_metadata=metadata,
                    is_master_candidate=len(active_sources) > 1,
                )
            )

        # Master candidates first, then descending by EBL LR
        suggestions.sort(
            key=lambda s: (
                -int(s.is_master_candidate),
                -(s.source_metadata.get("EBL", {}).get("lr", 0.0)),
            )
        )
        return suggestions

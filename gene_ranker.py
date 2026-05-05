"""gene_ranker.py

Ranks candidate genes within the top-ranked disease module.

Design decisions
----------------
Score formula:
    A(g, h) = 1_{h in G_h} + (1 - 1_{h in G_h}) * gamma * psi(g)
    score = sum(A(g, h) * p(h, m)) / sum(p(h, m) for h in observed)

    This Soft Module-Aware (SMA-GS) formula replaces binary annotation
    lookup with an affinity function. Genes receive full credit for
    direct annotations (exact match) and partial "leakage" credit for
    other module symptoms based on their coherence gate psi(g) and the
    global leakage parameter gamma.

    If observed is empty, score = 0 for all genes.

Coherence Gate psi(g):
    core       -> 1.0
    peripheral -> 0.5
    unstable   -> 0.0

Supporting phenotypes:
    The observed terms that the gene actually has annotated (exact matches),
    sorted descending by module prevalence. Leakage terms are excluded.

NPP slot:
    GeneCandidate.npp_score is set to None until npp_scores.csv is
    available. When it is, gene_ranker will be extended to include:
        score = lambda1 * phenotype_score + lambda2 * npp_score
"""

from __future__ import annotations

import pandas as pd

from output_models import GeneCandidate
from data_loader import DataLoader
from hpo_traversal import HPOTraversal

from ethnicity_prior_policy import EthnicityPriorPolicy, apply_ethnicity_policy

# Floor for the normalization denominator (prevents division by zero)
DENOM_FLOOR = 1e-9

DEFAULT_GAMMA = 0.3

_COHERENCE_GATE: dict[str, float] = {
    "core": 1.0,
    "peripheral": 0.5,
    "unstable": 0.0,
}


class GeneRanker:
    def __init__(
        self,
        data_loader: DataLoader,
        hpo_traversal: HPOTraversal,
        gamma: float = DEFAULT_GAMMA,
        lr_matrix: pd.DataFrame | None = None,
        count_matrix: pd.DataFrame | None = None,
        ebl_policy: EthnicityPriorPolicy | None = None,
    ) -> None:
        self._dl = data_loader
        self._ht = hpo_traversal
        self._gamma = gamma
        self._lr_matrix = lr_matrix
        self._count_matrix = count_matrix
        self._ebl_policy = ebl_policy or EthnicityPriorPolicy.default()

    def _coherence_gate(self, classification: str) -> float:
        """ψ(g): how representative this gene is of its module."""
        return _COHERENCE_GATE.get(classification, 0.0)

    def _compute_affinity(
        self,
        is_annotated: bool,
        psi: float,
    ) -> tuple[float, float]:
        """Return (exact_part, leak_part) of A(g, h).

        A(g, h) = exact_part + leak_part
                = 1_{h in G_h} + (1 - 1_{h in G_h}) * gamma * psi
        """
        if is_annotated:
            return 1.0, 0.0
        return 0.0, self._gamma * psi

    def rank_genes(
        self,
        module_id: int,
        observed: list[str],
        ethnicity_group: str | None = None,
        use_ethnicity_prior: bool | None = None,
    ) -> list[GeneCandidate]:
        """Rank all genes in module_id by phenotype overlap + stability.

        Parameters
        ----------
        module_id : the module whose genes are to be ranked
        observed  : list of HPO IDs present in the patient

        Returns
        -------
        List of GeneCandidate, sorted descending by score.
        """
        genes = self._dl.module_genes.get(module_id, [])
        observed_set = set(observed)

        # Pre-compute total prevalence mass of the query (denominator)
        total_obs_prev = sum(
            self._dl.get_prevalence(hpo, module_id)
            for hpo in observed_set
        )
        total_obs_prev = max(total_obs_prev, DENOM_FLOOR)

        candidates: list[GeneCandidate] = []

        for gene in genes:
            info = self._dl.gene_info[gene]
            gene_hpos = self._dl.gene_hpo.get(gene, set())

            classification = info["classification"]
            psi = self._coherence_gate(classification)

            breakdown: list[tuple[str, float]] = []
            leak_breakdown: list[tuple[str, float]] = []
            weighted_affinity = 0.0

            for hpo in observed_set:
                p = self._dl.get_prevalence(hpo, module_id)
                if p <= 0.0:
                    continue

                is_annotated = hpo in gene_hpos
                exact_part, leak_part = self._compute_affinity(is_annotated, psi)
                affinity = exact_part + leak_part
                if affinity <= 0.0:
                    continue

                contrib = affinity * p / total_obs_prev
                weighted_affinity += affinity * p

                term_name = self._dl.hpo_name.get(hpo, hpo)
                breakdown.append((term_name, contrib))

                if leak_part > 0.0:
                    leak_contrib = leak_part * p / total_obs_prev
                    leak_breakdown.append((term_name, leak_contrib))

            breakdown.sort(key=lambda x: x[1], reverse=True)
            leak_breakdown.sort(key=lambda x: x[1], reverse=True)

            phenotype_score = weighted_affinity / total_obs_prev

            # Evaluate ethnicity prior via policy layer
            decision = None
            if use_ethnicity_prior and ethnicity_group:
                 decision = apply_ethnicity_policy(
                     gene=gene,
                     ethnicity=ethnicity_group,
                     lr_matrix=self._lr_matrix,
                     count_matrix=self._count_matrix,
                     policy=self._ebl_policy
                 )

            # Default to no change if layer is off or decision not applied
            effective_lr = decision.effective_lr if decision else 1.0
            score = phenotype_score * effective_lr

            gate_contribution = sum(c for _, c in leak_breakdown)
            supporting_names = [
                name for name, _ in breakdown
                if name not in {n for n, _ in leak_breakdown}
            ]

            candidates.append(GeneCandidate(
                gene=gene,
                score=round(score, 6),
                stability=classification,
                supporting_phenotypes=supporting_names,
                npp_score=None,
                score_breakdown=breakdown,
                stability_breakdown=(classification, psi, round(gate_contribution, 6)),
                leak_breakdown=leak_breakdown,
                ethnicity_lr_raw=round(decision.raw_lr, 4) if decision else None,
                ethnicity_lr_effective=round(decision.effective_lr, 4) if decision else None,
                ethnicity_count_n=decision.count_n if decision else None,
                ethnicity_rule_reason=decision.reason_code if decision else None,
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

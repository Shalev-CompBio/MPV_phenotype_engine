"""gene_ranker.py

Ranks candidate genes within the top-ranked disease module.

Design decisions
----------------
Score formula:
    phenotype_score = sum(get_prevalence(hpo, module) for hpo in
                         (observed ∩ gene_hpo_profile))
                    / max(total_observed_prevalence, DENOM_FLOOR)

    This is a weighted Jaccard-like overlap: we credit each matching term
    by how prevalent it is in the module (more specific terms contribute
    more), normalized by the total prevalence mass of the query.

    If observed is empty, phenotype_score = 0 for all genes.

Stability modifier:
    stability_modifier = stability_score * STABILITY_WEIGHT * direction
    where direction:
        core       -> +1.0
        peripheral ->  0.0  (neutral; scored on phenotype match alone)
        unstable   -> -1.0

    STABILITY_WEIGHT = 0.2 keeps the modifier in [-0.2, +0.2] while
    phenotype_score is in [0, 1].  Core genes with no phenotype match
    still get a small positive lift; unstable genes are mildly penalized.

Final score:
    score = phenotype_score + stability_modifier

Supporting phenotypes:
    The observed terms that overlap with the gene's annotated HPO profile,
    sorted descending by module prevalence.

NPP slot:
    GeneCandidate.npp_score is set to None until npp_scores.csv is
    available.  When it is, gene_ranker will be extended to include:
        score = lambda1 * phenotype_score + lambda2 * npp_score
"""

from __future__ import annotations

from output_models import GeneCandidate
from data_loader import DataLoader
from hpo_traversal import HPOTraversal

# Weight applied to the stability modifier so it stays in [-0.2, 0.2]
STABILITY_WEIGHT = 0.2

# Floor for the normalization denominator (prevents division by zero)
DENOM_FLOOR = 1e-9

_STABILITY_DIRECTION: dict[str, float] = {
    "core": 1.0,
    "peripheral": 0.0,
    "unstable": -1.0,
}


class GeneRanker:
    def __init__(self, data_loader: DataLoader, hpo_traversal: HPOTraversal) -> None:
        self._dl = data_loader
        self._ht = hpo_traversal

    def rank_genes(
        self,
        module_id: int,
        observed: list[str],
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

            # Matching terms
            matching = observed_set & gene_hpos
            
            # Step 2 — Compute breakdown
            breakdown = []
            match_prev = 0.0
            for hpo in matching:
                p = self._dl.get_prevalence(hpo, module_id)
                match_prev += p
                contrib = p / total_obs_prev
                breakdown.append((self._dl.hpo_name.get(hpo, hpo), contrib))
            
            breakdown.sort(key=lambda x: x[1], reverse=True)
            phenotype_score = match_prev / total_obs_prev

            # Stability modifier
            classification = info["classification"]
            direction = _STABILITY_DIRECTION.get(classification, 0.0)
            stability_score = info["stability_score"]
            modifier_value = stability_score * STABILITY_WEIGHT * direction

            score = phenotype_score + modifier_value

            # Supporting phenotypes: matching HPO terms, sorted by module prevalence
            supporting_names = [name for name, _ in breakdown]

            candidates.append(GeneCandidate(
                gene=gene,
                score=round(score, 6),
                stability=classification,
                supporting_phenotypes=supporting_names,
                npp_score=None,
                score_breakdown=breakdown,
                stability_breakdown=(classification, stability_score, modifier_value),
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

"""scoring_engine.py

Core Naive Bayes computation over 17 disease modules.

Design decisions
----------------
Ancestor expansion (observed terms only):
    Excluding a term does not confirm its parents are absent (a specific
    retinal dystrophy being absent doesn't rule out retinal disease in
    general).  Therefore ancestor expansion is applied only to observed
    terms, not to excluded terms.

Term weight aggregation:
    When multiple observed terms share ancestors, we keep the *maximum*
    weight for each ancestor (not a sum), to avoid double-counting
    confirmation of the same high-level term.

Log-likelihood clipping:
    Prevalences are clipped to [EPSILON, 1 - EPSILON] before taking logs
    to avoid log(0) = -inf.

Prior:
    Uniform over 17 modules by default (log_prior = 0 for all modules).
    The prior is left as a hook for future module-size weighting.

Confidence:
    Normalized entropy: 1 - H(posterior) / log2(17)
    0 = maximum uncertainty (flat posterior), 1 = all mass on one module.
"""

from __future__ import annotations

import math
from data_loader import DataLoader, MODULE_IDS
from hpo_traversal import HPOTraversal

# Prevalence bounds for numerical stability in log computations
EPSILON = 0.001
ONE_MINUS_EPSILON = 1.0 - EPSILON

_LOG2_17 = math.log2(17)


class ScoringEngine:
    def __init__(self, data_loader: DataLoader, hpo_traversal: HPOTraversal) -> None:
        self._dl = data_loader
        self._ht = hpo_traversal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_modules(
        self,
        observed: list[str],
        excluded: list[str],
    ) -> dict[int, float]:
        """Compute posterior probability over 17 modules.

        Parameters
        ----------
        observed : list of HPO IDs the patient has
        excluded : list of HPO IDs explicitly absent

        Returns
        -------
        dict mapping module_id -> probability (sums to 1.0)
        """
        # Build weighted term map for all observed terms + their ancestors.
        # For each HPO ID, keep the maximum weight seen (direct = 1.0).
        obs_weights: dict[str, float] = {}
        for hpo_id in observed:
            obs_weights[hpo_id] = max(obs_weights.get(hpo_id, 0.0), 1.0)
            for anc_id, w in self._ht.get_ancestors(hpo_id):
                obs_weights[anc_id] = max(obs_weights.get(anc_id, 0.0), w)

        # Compute log-likelihood for each module
        log_scores: dict[int, float] = {}
        for module_id in MODULE_IDS:
            ll = 0.0

            # Observed term contributions (weighted)
            for hpo_id, weight in obs_weights.items():
                p = self._dl.get_prevalence(hpo_id, module_id)
                p = max(min(p, ONE_MINUS_EPSILON), EPSILON)
                ll += weight * math.log(p)

            # Excluded term contributions (direct only, no ancestor expansion)
            for hpo_id in excluded:
                p = self._dl.get_prevalence(hpo_id, module_id)
                p = max(min(p, ONE_MINUS_EPSILON), EPSILON)
                ll += math.log(1.0 - p)

            # Uniform log-prior: 0 (log(1/17) cancels in normalisation)
            log_scores[module_id] = ll

        return _softmax(log_scores)

    def compute_confidence(self, posterior: dict[int, float]) -> float:
        """Normalized entropy: 1 - H(posterior) / log2(17).

        0 = maximum uncertainty, 1 = complete certainty.
        """
        probs = [posterior.get(m, 0.0) for m in MODULE_IDS]
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        return 1.0 - entropy / _LOG2_17


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _softmax(log_scores: dict[int, float]) -> dict[int, float]:
    """Numerically stable softmax over log-likelihoods."""
    max_ll = max(log_scores.values())
    exp_scores = {m: math.exp(ll - max_ll) for m, ll in log_scores.items()}
    total = sum(exp_scores.values())
    return {m: v / total for m, v in exp_scores.items()}

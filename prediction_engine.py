"""prediction_engine.py

Phenotype predictions and next-question suggestions.

Design decisions
----------------
Phenotype prediction thresholds:
    recommended_workup : prevalence >= 0.50 in the matched module
        These are common findings; the clinician should actively look for them.
    prognostic_risk    : 0.15 <= prevalence < 0.50
        Less certain but clinically meaningful; worth flagging.

    Already-observed terms are excluded from both lists.
    Each list is capped at MAX_TERMS (20) and sorted descending by prevalence.

Children expansion (downward traversal):
    For each term in the output lists, get_children() is called to retrieve
    specific sub-terms present in the IRD annotation space.  These are
    appended after their parent; duplicates across parents are skipped.
    This allows the clinician to see more granular phenotype options.

Next-question candidate pool:
    Terms with prevalence >= IG_PREVALENCE_THRESHOLD (0.05) in ANY module
    and not already asked.  If this pool exceeds MAX_IG_CANDIDATES (300),
    we retain the 300 terms with the highest maximum-module prevalence.
    The IG computation calls score_modules twice per candidate (yes/no
    simulation); 300 candidates × 2 × 17 modules is fast.

Marginal probability of "yes" for a candidate term q:
    p_yes = sum(posterior[m] * get_prevalence(q, m) for m in all modules)
    This is the expected prevalence under the current posterior belief.
"""

from __future__ import annotations

import math

from output_models import HPOTerm, PhenotypePrediction, SuggestedQuestion
from data_loader import DataLoader, MODULE_IDS
from hpo_traversal import HPOTraversal
from scoring_engine import ScoringEngine

# Thresholds for phenotype prediction buckets
WORKUP_THRESHOLD = 0.50
RISK_THRESHOLD = 0.15

# Maximum terms per output category (prevents overwhelming output)
MAX_TERMS = 20
MAX_PROGRESSION_TERMS = 10  # cap for likely_next_manifestations

# IG candidate selection
IG_PREVALENCE_THRESHOLD = 0.05
MAX_IG_CANDIDATES = 300


class PredictionEngine:
    def __init__(
        self,
        data_loader: DataLoader,
        hpo_traversal: HPOTraversal,
        scoring_engine: ScoringEngine,
    ) -> None:
        self._dl = data_loader
        self._ht = hpo_traversal
        self._se = scoring_engine

        # Pre-build the IG candidate pool: all IRD terms with prevalence >= threshold
        # in at least one module.  Stored as (hpo_id, max_prevalence) for fast sorting.
        self._ig_candidates: list[tuple[str, float]] = self._build_ig_pool()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_phenotypes(
        self,
        module_id: int,
        observed: list[str],
    ) -> PhenotypePrediction:
        """Return recommended workup and prognostic risk phenotypes.

        Parameters
        ----------
        module_id : the top-ranked disease module
        observed  : HPO IDs already known to be present
        """
        observed_set = set(observed)

        # All ancestors of observed terms — these are ontologically subsumed
        # by the observed findings and add no clinical specificity.
        ancestors_of_observed: set[str] = set()
        for obs_id in observed:
            for anc_id, _ in self._ht.get_ancestors(obs_id):
                ancestors_of_observed.add(anc_id)

        # Terms with background >= GENERIC_THRESHOLD are root-level taxonomy nodes
        # (e.g., "Phenotypic abnormality") present in every module and not
        # actionable as clinical findings.
        GENERIC_THRESHOLD = 0.80

        # Gather all terms in the module sorted by prevalence (descending)
        module_terms = sorted(
            (
                (hpo_id, self._dl.get_prevalence(hpo_id, module_id))
                for hpo_id in self._dl.ird_terms
                if self._dl.get_prevalence(hpo_id, module_id) > 0
                and self._dl.get_background(hpo_id) < GENERIC_THRESHOLD
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        workup_ids: list[str] = []
        risk_ids: list[str] = []
        seen: set[str] = set(observed_set)

        for hpo_id, prev in module_terms:
            if hpo_id in seen:
                continue
            if hpo_id in ancestors_of_observed:   # NEW
                continue
            if prev >= WORKUP_THRESHOLD:
                if len(workup_ids) < MAX_TERMS:
                    workup_ids.append(hpo_id)
                    seen.add(hpo_id)
            elif prev >= RISK_THRESHOLD:
                if len(risk_ids) < MAX_TERMS:
                    risk_ids.append(hpo_id)
                    seen.add(hpo_id)

        # Expand each term with its IRD-space children
        workup = self._expand_with_children(workup_ids, module_id)
        risk = self._expand_with_children(risk_ids, module_id)

        # Likely Next Manifestations: depth-1 HPO children of observed terms
        # with any module prevalence. No minimum threshold — clinical relevance
        # comes from the ontology relationship. Deduplicated against workup/risk.
        workup_risk_ids: set[str] = (
            {t.hpo_id for t in workup} | {t.hpo_id for t in risk}
        )
        next_candidates: list[tuple[float, str]] = []
        seen_next: set[str] = set()
        for obs_id in observed:
            for child_id in self._ht.get_children(obs_id, depth=1):
                if child_id in seen_next:
                    continue
                seen_next.add(child_id)
                if child_id in observed_set or child_id in workup_risk_ids:
                    continue
                child_prev = self._dl.get_prevalence(child_id, module_id)
                if child_prev > 0:
                    next_candidates.append((child_prev, child_id))

        next_candidates.sort(key=lambda x: x[0], reverse=True)
        likely_next: list[HPOTerm] = [
            HPOTerm(
                hpo_id=cid,
                term_name=self._dl.hpo_name.get(cid, cid),
                prevalence=cprev,
            )
            for cprev, cid in next_candidates[:MAX_PROGRESSION_TERMS]
        ]

        return PhenotypePrediction(
            recommended_workup=workup,
            prognostic_risk=risk,
            likely_next_manifestations=likely_next,
        )

    def suggest_next_question(
        self,
        posterior: dict[int, float],
        observed: list[str],
        excluded: list[str],
    ) -> SuggestedQuestion:
        """Return the HPO term that would maximally reduce posterior entropy.

        Parameters
        ----------
        posterior : current module probability distribution
        observed  : HPO IDs already answered 'yes'
        excluded  : HPO IDs already answered 'no'
        """
        qs = self.suggest_next_questions(posterior, observed, excluded, k=1)
        if qs:
            return qs[0]
        fb = "HP:0000510"
        return SuggestedQuestion(
            hpo_id=fb,
            term_name=self._dl.hpo_name.get(fb, fb),
            information_gain=0.0,
        )

    def suggest_next_questions(
        self,
        posterior: dict[int, float],
        observed: list[str],
        excluded: list[str],
        k: int = 5,
    ) -> list[SuggestedQuestion]:
        """Same information-gain scoring as suggest_next_question; return top *k* terms."""
        asked: set[str] = set(observed) | set(excluded)
        k = max(1, k)

        if not observed and not excluded:
            out: list[SuggestedQuestion] = []
            for hpo_id, _ in self._ig_candidates:
                if hpo_id not in asked:
                    out.append(
                        SuggestedQuestion(
                            hpo_id=hpo_id,
                            term_name=self._dl.hpo_name.get(hpo_id, hpo_id),
                            information_gain=0.0,
                        )
                    )
                    if len(out) >= k:
                        break
            return out

        candidates = [
            (hpo_id, max_prev)
            for hpo_id, max_prev in self._ig_candidates
            if hpo_id not in asked
        ]
        if len(candidates) > MAX_IG_CANDIDATES:
            candidates = candidates[:MAX_IG_CANDIDATES]

        current_h = _entropy(list(posterior.values()))
        scored: list[tuple[float, str]] = []

        for hpo_id, _ in candidates:
            p_yes = sum(
                posterior.get(m, 0.0) * self._dl.get_prevalence(hpo_id, m)
                for m in MODULE_IDS
            )
            p_yes = max(min(p_yes, 0.9999), 0.0001)
            p_no = 1.0 - p_yes

            posterior_yes = self._se.score_modules(observed + [hpo_id], excluded)
            posterior_no = self._se.score_modules(observed, excluded + [hpo_id])

            h_yes = _entropy(list(posterior_yes.values()))
            h_no = _entropy(list(posterior_no.values()))

            ig = current_h - (p_yes * h_yes + p_no * h_no)
            scored.append((ig, hpo_id))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[SuggestedQuestion] = []
        seen: set[str] = set()
        for ig, hpo_id in scored:
            if hpo_id in seen:
                continue
            seen.add(hpo_id)
            out.append(
                SuggestedQuestion(
                    hpo_id=hpo_id,
                    term_name=self._dl.hpo_name.get(hpo_id, hpo_id),
                    information_gain=round(ig, 6),
                )
            )
            if len(out) >= k:
                break

        if not out and candidates:
            h0 = candidates[0][0]
            out.append(
                SuggestedQuestion(
                    hpo_id=h0,
                    term_name=self._dl.hpo_name.get(h0, h0),
                    information_gain=0.0,
                )
            )

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_ig_pool(self) -> list[tuple[str, float]]:
        """Build sorted list of (hpo_id, max_prevalence) for IG computation."""
        pool: dict[str, float] = {}
        for hpo_id in self._dl.ird_terms:
            max_prev = max(
                self._dl.get_prevalence(hpo_id, m)
                for m in MODULE_IDS
            )
            if max_prev >= IG_PREVALENCE_THRESHOLD:
                pool[hpo_id] = max_prev
        # Sort descending by max prevalence; most discriminating terms first
        return sorted(pool.items(), key=lambda x: x[1], reverse=True)

    def _expand_with_children(
        self,
        hpo_ids: list[str],
        module_id: int,
    ) -> list[HPOTerm]:
        """Return HPOTerm list for given IDs, each followed by its IRD children."""
        result: list[HPOTerm] = []
        seen: set[str] = set()

        for hpo_id in hpo_ids:
            if hpo_id in seen:
                continue
            seen.add(hpo_id)
            result.append(HPOTerm(
                hpo_id=hpo_id,
                term_name=self._dl.hpo_name.get(hpo_id, hpo_id),
                prevalence=self._dl.get_prevalence(hpo_id, module_id),
            ))
            for child_id in self._ht.get_children(hpo_id, depth=1):
                if child_id not in seen:
                    child_prev = self._dl.get_prevalence(child_id, module_id)
                    # Only include children actually present in this module
                    if child_prev > 0:
                        seen.add(child_id)
                        result.append(HPOTerm(
                            hpo_id=child_id,
                            term_name=self._dl.hpo_name.get(child_id, child_id),
                            prevalence=child_prev,
                        ))

        return result


def ig_qualitative_label(ig: float) -> str:
    """Map a raw information-gain value (in nats) to a three-tier qualitative label.

    Thresholds are calibrated against the maximum possible entropy of the
    17-module posterior: H_max = ln(17) ≈ 2.833 nats.

    Tiers
    -----
    High diagnostic value    : ig >= 0.8 nats  (~28% of H_max)
    Moderate diagnostic value: 0.3 <= ig < 0.8
    Low diagnostic value     : ig  < 0.3

    These defaults can be adjusted after empirical calibration.
    """
    if ig >= 0.8:
        return "High diagnostic value"
    if ig >= 0.3:
        return "Moderate diagnostic value"
    return "Low diagnostic value"


def _entropy(probs: list[float]) -> float:
    """Shannon entropy in nats."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h

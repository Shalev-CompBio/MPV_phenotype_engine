"""session_manager.py

Stateful interactive session for step-by-step Q&A diagnosis.

Design decisions
----------------
A Session wraps the full engine stack and maintains two growing lists:
  - observed  : HPO IDs answered 'yes'
  - excluded  : HPO IDs answered 'no'

After each answer, the posterior is recomputed lazily on the next call to
get_current_result() or get_next_question(), not eagerly on every answer.
This keeps answer_yes/answer_no O(1) and the score only computed when needed.

The session does not store the full QueryResult after each step — it only
stores the raw lists and recomputes on demand.  This avoids stale cached
results if someone peeks at the state mid-session.
"""

from __future__ import annotations

from output_models import QueryResult
from data_loader import DataLoader
from scoring_engine import ScoringEngine
from gene_ranker import GeneRanker
from prediction_engine import PredictionEngine


class Session:
    """A single interactive Q&A session."""

    def __init__(
        self,
        data_loader: DataLoader,
        scoring_engine: ScoringEngine,
        gene_ranker: GeneRanker,
        prediction_engine: PredictionEngine,
    ) -> None:
        self._dl = data_loader
        self._se = scoring_engine
        self._gr = gene_ranker
        self._pe = prediction_engine

        self.observed: list[str] = []
        self.excluded: list[str] = []

    def answer_yes(self, hpo_id: str) -> None:
        """Record that the patient HAS this phenotype."""
        hpo_id = hpo_id.strip().upper()
        if hpo_id not in self.observed:
            self.observed.append(hpo_id)
        # Remove from excluded if previously marked absent
        if hpo_id in self.excluded:
            self.excluded.remove(hpo_id)

    def answer_no(self, hpo_id: str) -> None:
        """Record that the patient does NOT have this phenotype."""
        hpo_id = hpo_id.strip().upper()
        if hpo_id not in self.excluded:
            self.excluded.append(hpo_id)
        if hpo_id in self.observed:
            self.observed.remove(hpo_id)

    def get_next_question(self):
        """Return the next most informative HPO question as a SuggestedQuestion."""
        posterior = self._se.score_modules(self.observed, self.excluded)
        return self._pe.suggest_next_question(posterior, self.observed, self.excluded)

    def get_current_result(self) -> QueryResult:
        """Compute and return the full QueryResult for the current session state."""
        from clinical_support import _build_result  # imported here to avoid circular import
        return _build_result(
            self._dl, self._se, self._gr, self._pe,
            self.observed, self.excluded,
        )

    def reset(self) -> None:
        """Clear all answers and restart the session."""
        self.observed = []
        self.excluded = []

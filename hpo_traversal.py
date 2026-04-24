"""hpo_traversal.py

All HPO tree operations.  A single HPOTraversal object is created at startup
and shared across the system.

Design decisions
----------------
Ancestor weights (upward traversal):
    distance 1 (direct parent)  -> 0.80
    distance 2 (grandparent)    -> 0.60
    distance 3                  -> 0.40
    distance 4+                 -> 0.20  (floor)
  This gives a fast, linear decay that never reaches zero.

Ancestor de-duplication:
    When multiple observed terms share ancestors, we keep the *maximum* weight
    seen for each ancestor (not a sum), to avoid double-counting.  The caller
    is responsible for merging results across multiple terms (see scoring_engine).

Children filter:
    get_children returns only HPO IDs present in the IRD annotation space
    (i.e., terms that appear in at least one of the 17 module sheets).  Terms
    outside this space are not informative for module discrimination.

HPO loading note:
    pronto emits a UnicodeWarning on the obo file; this is harmless (the
    content is read correctly) and is suppressed here via warnings.filterwarnings.
"""

from __future__ import annotations

import urllib.request
import warnings
from pathlib import Path
from collections import deque

import pronto

_HP_OBO_URL = "https://purl.obolibrary.org/obo/hp.obo"


def _ensure_hp_obo(obo_path: Path) -> None:
    """Download hp.obo from OBO Foundry if it is missing locally.

    The file is ~10 MB and is excluded from the repository (.gitignore).
    On first run the download takes ~30 s depending on network speed.
    """
    if obo_path.exists():
        return
    obo_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[hpo_traversal] hp.obo not found — downloading from {_HP_OBO_URL} …")
    with urllib.request.urlopen(_HP_OBO_URL, timeout=120) as resp, open(obo_path, "wb") as fh:
        fh.write(resp.read())
    print(f"[hpo_traversal] hp.obo saved to {obo_path}")

# Ancestor weight parameters
_WEIGHT_START = 0.80
_WEIGHT_STEP = 0.20
_WEIGHT_FLOOR = 0.20


class HPOTraversal:
    def __init__(self, obo_path: str | Path, ird_terms: frozenset[str]) -> None:
        """
        Parameters
        ----------
        obo_path : path to hp.obo
        ird_terms : frozenset of HPO IDs in the IRD annotation space
                    (from DataLoader.ird_terms)
        """
        obo_path = Path(obo_path)
        _ensure_hp_obo(obo_path)  # no-op if file already exists
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ont = pronto.Ontology(str(obo_path))
        self._ird_terms = ird_terms

        # Pre-build name -> id lookup from ontology for term resolution
        self._ont_name_to_id: dict[str, str] = {
            t.name.lower(): t.id
            for t in self._ont.terms()
            if t.name
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ancestors(self, hpo_id: str) -> list[tuple[str, float]]:
        """Return all ancestors of hpo_id with decreasing weights.

        Returns a list of (ancestor_hpo_id, weight) tuples, ordered by
        increasing distance.  The term itself is NOT included.
        If hpo_id is unknown to the ontology, returns an empty list.
        """
        if hpo_id not in self._ont:
            return []

        results: list[tuple[str, float]] = []
        seen: set[str] = {hpo_id}
        # BFS by layer; each layer is one step further from the query term
        current_layer: set[str] = {hpo_id}
        distance = 0

        while current_layer:
            distance += 1
            weight = max(_WEIGHT_START - _WEIGHT_STEP * (distance - 1), _WEIGHT_FLOOR)
            next_layer: set[str] = set()

            for tid in current_layer:
                if tid not in self._ont:
                    continue
                # superclasses(distance=1) gives immediate parents only
                for parent in self._ont[tid].superclasses(distance=1, with_self=False):
                    if parent.id not in seen:
                        seen.add(parent.id)
                        next_layer.add(parent.id)
                        results.append((parent.id, weight))

            current_layer = next_layer

        return results

    def get_children(self, hpo_id: str, depth: int = 1) -> list[str]:
        """Return HPO IDs that are direct children of hpo_id.

        Only returns terms present in the IRD annotation space.
        depth > 1 is supported (BFS up to that depth), but the SPEC
        currently only calls this with depth=1.
        """
        if hpo_id not in self._ont:
            return []

        results: list[str] = []
        seen: set[str] = {hpo_id}
        queue: deque[tuple[str, int]] = deque([(hpo_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            if current_id not in self._ont:
                continue
            for child in self._ont[current_id].subclasses(distance=1, with_self=False):
                if child.id not in seen:
                    seen.add(child.id)
                    if child.id in self._ird_terms:
                        results.append(child.id)
                    queue.append((child.id, current_depth + 1))

        return results

    def resolve(self, query: str) -> str | None:
        """Resolve a free-text name or HP:XXXXXXX string to a canonical HPO ID.

        Returns None if resolution fails.
        """
        query = query.strip()
        # Already an HPO ID
        if query.upper().startswith("HP:") and query.upper() in self._ont:
            return query.upper()
        # Case-insensitive name match against ontology
        lower = query.lower()
        if lower in self._ont_name_to_id:
            return self._ont_name_to_id[lower]
        return None

    def term_name(self, hpo_id: str) -> str:
        """Return the preferred term name from the ontology, or the ID itself."""
        if hpo_id in self._ont:
            return self._ont[hpo_id].name or hpo_id
        return hpo_id

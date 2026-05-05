from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModuleMatch:
    module_id: int
    module_label: str | None        # clinical label - empty by default, to be filled manually
    probability: float
    supporting_phenotypes: list[str]


@dataclass
class GeneCandidate:
    gene: str
    score: float
    stability: str                  # "core" | "peripheral" | "unstable"
    supporting_phenotypes: list[str]
    npp_score: float | None         # None until NPP data available
    score_breakdown: list[tuple[str, float]] = field(default_factory=list)
    stability_breakdown: tuple[str, float, float] | None = None
    leak_breakdown: list[tuple[str, float]] = field(default_factory=list)
    ethnicity_lr_raw: float | None = None       # EBL raw LR
    ethnicity_lr_effective: float | None = None # applied EBL multiplier; None = layer off
    ethnicity_count_n: float | None = None      # count of training cases
    ethnicity_rule_reason: str | None = None    # transparent rule application reason


@dataclass
class DiscoverySuggestion:
    """Track 2: a gene absent from the primary module set with significant ethnic signal."""
    gene: str
    sources: list[str]              # e.g. ["EBL"]
    source_metadata: dict           # e.g. {"EBL": {"lr": 3.38, "n": 15, "ethnicity": "North_African_Jewish"}}
    is_master_candidate: bool = False   # True when ≥2 evidence sources agree


@dataclass
class HPOTerm:
    hpo_id: str
    term_name: str
    prevalence: float | None = None


@dataclass
class PhenotypePrediction:
    recommended_workup: list[HPOTerm]
    prognostic_risk: list[HPOTerm]
    likely_next_manifestations: list[HPOTerm] = field(default_factory=list)


@dataclass
class SuggestedQuestion:
    hpo_id: str
    term_name: str
    information_gain: float


@dataclass
class QueryResult:
    top_module: ModuleMatch
    all_modules: list[ModuleMatch]
    candidate_genes: list[GeneCandidate]
    phenotype_predictions: PhenotypePrediction
    next_question: SuggestedQuestion
    confidence: float
    next_questions: list[SuggestedQuestion] = field(default_factory=list)

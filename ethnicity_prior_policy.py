from dataclasses import dataclass
import pandas as pd


@dataclass
class EthnicityPriorPolicy:
    """
    Configuration for how the Ethnicity Bayes Layer applies its likelihood ratios.
    Centralizing this prevents hardcoded thresholds from scattering across the codebase.
    """
    mode: str = "upweight_only"  # Options: "upweight_only", "symmetric", "disabled"
    min_n_upweight: int = 5      # Minimum training cases needed to apply a boost
    min_lr_upweight: float = 2.0 # Minimum LR needed to apply a boost
    min_n_partial_upweight: int = 7   # Minimum cases needed for partial boost lane
    min_lr_partial_upweight: float = 1.4  # Minimum LR needed for partial boost lane
    partial_upweight_strength: float = 0.5  # Apply 50% of the LR effect in partial lane
    allow_downweight: bool = False # If True, enables penalizing genes (requires symmetric mode)

    @classmethod
    def default(cls) -> "EthnicityPriorPolicy":
        return cls()


@dataclass
class EthnicityPriorDecision:
    """
    The transparent outcome of applying the ethnicity policy to a single gene.
    """
    raw_lr: float
    effective_lr: float
    count_n: float
    applied: bool
    reason_code: str


def apply_ethnicity_policy(
    gene: str,
    ethnicity: str,
    lr_matrix: pd.DataFrame | None,
    count_matrix: pd.DataFrame | None,
    policy: EthnicityPriorPolicy
) -> EthnicityPriorDecision:
    """
    Evaluates the LR and training count for a given gene/ethnicity against the policy,
    returning the effective multiplier to use.

    Safe defaults: returns effective_lr=1.0 if data is missing or policy prevents action.
    """
    if policy.mode == "disabled":
        return EthnicityPriorDecision(1.0, 1.0, 0.0, False, "policy_disabled")

    if not ethnicity or lr_matrix is None or count_matrix is None:
        return EthnicityPriorDecision(1.0, 1.0, 0.0, False, "missing_data")

    if gene not in lr_matrix.index or ethnicity not in lr_matrix.columns:
        return EthnicityPriorDecision(1.0, 1.0, 0.0, False, "gene_or_ethnicity_unseen")

    # Extract values
    raw_lr = float(lr_matrix.at[gene, ethnicity])
    count_n = float(count_matrix.at[gene, ethnicity]) if gene in count_matrix.index else 0.0

    # Handle NaN or non-positive LRs (data errors)
    if pd.isna(raw_lr) or raw_lr <= 0:
         return EthnicityPriorDecision(1.0, 1.0, count_n, False, "invalid_lr")

    # Evaluate against policy
    if policy.mode == "upweight_only":
        if count_n >= policy.min_n_upweight and raw_lr >= policy.min_lr_upweight:
            return EthnicityPriorDecision(raw_lr, raw_lr, count_n, True, "boost_applied")
        if (
            count_n >= policy.min_n_partial_upweight
            and policy.min_lr_partial_upweight <= raw_lr < policy.min_lr_upweight
        ):
            effective_lr = 1.0 + policy.partial_upweight_strength * (raw_lr - 1.0)
            return EthnicityPriorDecision(raw_lr, effective_lr, count_n, True, "partial_boost_applied")
        if raw_lr >= policy.min_lr_upweight:
            return EthnicityPriorDecision(raw_lr, 1.0, count_n, False, f"insufficient_evidence_n_lt_{policy.min_n_upweight}")
        if raw_lr < 1.0:
            return EthnicityPriorDecision(raw_lr, 1.0, count_n, False, "downweight_prevented_by_policy")
        if raw_lr >= policy.min_lr_partial_upweight:
            return EthnicityPriorDecision(raw_lr, 1.0, count_n, False, f"insufficient_evidence_partial_n_lt_{policy.min_n_partial_upweight}")
        return EthnicityPriorDecision(raw_lr, 1.0, count_n, False, "lr_below_boost_threshold")

    # Future modes (e.g., symmetric) would be implemented here

    # Fallback safety net
    return EthnicityPriorDecision(raw_lr, 1.0, count_n, False, "unhandled_policy_path")

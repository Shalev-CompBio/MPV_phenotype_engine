# MPV Phenotype Engine

A probabilistic clinical decision-support system for Inherited Retinal Disease (IRD) diagnosis.

This engine accepts partial phenotype evidence (HPO terms observed and/or excluded) and returns:
- Posterior probability over 17 IRD disease modules
- Ranked candidate genes within the top module
- Active-learning next questions (information gain)
- Predicted phenotypes for workup and prognosis
- HPO hierarchy-aware likely next manifestations

---

## 1) Current Status (Initial Updated README)

This document is an updated initial technical README aligned to the current implementation in:
- `data_loader.py`
- `hpo_traversal.py`
- `scoring_engine.py`
- `gene_ranker.py`
- `prediction_engine.py`
- `clinical_support.py`
- `session_manager.py`
- `app.py`

It emphasizes:
- Complete logic and pipeline steps
- Precise description of produced results
- Current validation outcomes from the codebase

---

## 2) End-to-End Pipeline (Complete Process)

### 2.1 Input Layer

The system ingests three core data sources at startup:
- Module prevalence matrix (`module_all_HPO_background_comparison_20260413_1019.xlsx`, 17 sheets)
- Gene/module/stability assignments (`gene_classification_20260412_1524.csv`)
- HPO ontology graph (`hp.obo`, parsed with `pronto`)

Optional analytic/support files are also loaded when present:
- Module signature file (`module_phenotypic_signatures_FDR_corrected_20260412_1524.csv`)
- Comparative analytics files under `Input/IRD_vs_Non-IRD/`, `Input/modules_RetiGene_Comparison/`, and `Input/comparative_signatures_all_IRD/`

### 2.2 Index Construction

`DataLoader` builds the following indexes:
- `(hpo_id, module_id) -> prevalence_fraction`
- `hpo_id -> background_prevalence` (mean over 17 modules)
- `hpo_id -> phenotype_name`
- `name_lower -> hpo_id`
- `module_id -> genes[]`
- `gene -> {module_id, stability_score, classification}`
- `gene -> set(hpo_ids)` (inverted from module sheets)
- `module_id -> significant_signature_terms[]` (if signature CSV exists)
- `hpo_id -> set(module_ids)` for signature highlighting

Fallback behavior for prevalence lookup:
1. exact `(hpo_id, module_id)` if available
2. module-agnostic background mean for `hpo_id`
3. global floor `0.001`

### 2.3 Ontology Traversal Layer

`HPOTraversal` provides:
- Upward traversal: ancestors with distance-decaying weights
- Downward traversal: children filtered to IRD annotation space (`ird_terms`)
- Free-text term resolution to canonical HPO IDs

Ancestor weight schedule:
- distance 1: 0.80
- distance 2: 0.60
- distance 3: 0.40
- distance >= 4: 0.20 (floor)

### 2.4 Probabilistic Scoring Layer

`ScoringEngine.score_modules(observed, excluded)` executes:

1. Build weighted observed map:
- direct observed term weight = 1.0
- add ancestor terms with decay weights
- if repeated, keep max weight (not sum)

2. Compute module log-likelihood:

$$
\log L(m) = \sum_{h \in O_w} w_h \log(\tilde p_{h,m}) + \sum_{h \in E} \log(1-\tilde p_{h,m})
$$

where:
- $O_w$: weighted observed map (observed + ancestors)
- $E$: excluded terms (direct only)
- $\tilde p_{h,m}$ is clipped to $[0.001, 0.999]$

3. Normalize with numerically stable softmax to get posterior over 17 modules.

Important modeling decision:
- Ancestor expansion is applied only to observed terms, not excluded terms.

### 2.5 Confidence Layer

Confidence is normalized entropy complement:

$$
confidence = 1 - \frac{H(p)}{\log_2(17)}
$$

with:

$$
H(p) = -\sum_i p_i\log_2 p_i
$$

Interpretation:
- 0.0 = maximal uncertainty (flat posterior)
- 1.0 = maximal certainty (all mass on one module)

### 2.6 Gene Ranking Layer

`GeneRanker.rank_genes(top_module, observed)` computes for each gene in the top module:

1. Phenotype overlap score:

$$
phenotypeScore(g)=\frac{\sum_{h \in (O \cap G_h)} p_{h,m}}{\max(\sum_{h \in O} p_{h,m}, 10^{-9})}
$$

2. Stability modifier:

$$
modifier=stabilityScore * 0.2 * d(c)
$$

with direction:
- `core`: +1
- `peripheral`: 0
- `unstable`: -1

3. Final gene score:

$$
score=phenotypeScore+modifier
$$

Outputs also include:
- matching phenotype names
- per-term contribution breakdown (`score_breakdown`)
- stability breakdown tuple (`stability_breakdown`)
- `npp_score` placeholder (`None` in current release)

### 2.7 Prediction + Active Learning Layer

`PredictionEngine` returns:

1. `recommended_workup`
- terms with module prevalence >= 50%
- excluding already observed and ancestors of observed
- generic terms filtered by background prevalence >= 80%

2. `prognostic_risk`
- terms with prevalence in [15%, 50%)

3. `likely_next_manifestations`
- depth-1 children of observed terms
- must be in IRD space and have module prevalence > 0
- deduplicated vs workup/risk

Information gain question selection (`suggest_next_questions`):
- candidate pool prebuilt from terms with max prevalence >= 5% in any module
- pool capped at 300
- exclude already asked terms
- for each candidate $q$:

$$
p_{yes} = \sum_m p(m)\,P(q|m),\quad p_{no}=1-p_{yes}
$$

$$
IG(q)=H(p)-\left[p_{yes}H(p|q{=}yes)+p_{no}H(p|q{=}no)\right]
$$

Entropy here is computed in nats.

Qualitative IG labels in UI:
- High diagnostic value: >= 0.8 nats
- Moderate diagnostic value: 0.3 to <0.8 nats
- Low diagnostic value: < 0.3 nats

### 2.8 Orchestration Layer

`ClinicalSupportEngine` assembles the full `QueryResult`:
1. score modules
2. compute confidence
3. choose top module
4. rank genes within top module
5. predict workup/risk/next manifestations
6. suggest top-5 next questions

This is exposed via:
- `query(observed, excluded)`
- `query_gene(gene_symbol)`
- `new_session()` (stateful Q&A loop)

### 2.9 UI/Delivery Layer

`app.py` provides four modes:
- Phenotype Query
- Interactive Session
- Module Browser
- Comparative Analytics

UI-specific behavior includes:
- add-to-query reactive rerun from workup/risk/next lists
- session history with Undo/Flip
- downloadable gene CSV and optional PDF summary
- module signature highlighting in phenotype outputs

---

## 3) Results: What the Engine Produces

For each run, the engine returns a complete `QueryResult` with:

1. Top module and full posterior distribution (`all_modules`, all 17)
2. Confidence score (entropy-normalized)
3. Ranked candidate genes (top module)
4. Phenotype prediction bundles:
- recommended workup
- prognostic risk
- likely next manifestations
5. Active-learning suggestions (top 5 questions by information gain)

### 3.1 Result Semantics (Clinical/Technical)

- Module probability: posterior belief conditioned on observed/excluded evidence under Naive Bayes assumptions.
- Confidence: sharpness of posterior, not external calibration probability.
- Gene score: hybrid of phenotype overlap and cluster-stability prior.
- IG score: expected reduction in posterior entropy if the phenotype is queried next.

### 3.2 Output Model (Current)

Primary dataclasses from `output_models.py`:
- `ModuleMatch`
- `GeneCandidate` (includes `score_breakdown` and `stability_breakdown`)
- `HPOTerm`
- `PhenotypePrediction`
- `SuggestedQuestion`
- `QueryResult`

---

## 4) Validation Results (Current Run)

Validation was executed using the classic case suite defined in `validation/classic_cases.py`.

Summary:
- Passed: 9/9
- Top-ranked module matched expected module in all cases

Detailed results:

| Case | Expected Module | Top Module | Rank of Expected | Top Probability |
|---|---:|---:|---:|---:|
| Bardet-Biedl | 9 | 9 | 1 | 0.999683 |
| Usher Type 1 | 16 | 16 | 1 | 0.983325 |
| Achromatopsia | 15 | 15 | 1 | 1.000000 |
| CSNB | 6 | 6 | 1 | 0.992260 |
| Isolated RP | 0 | 0 | 1 | 0.802411 |
| Choroideremia | 0 | 0 | 1 | 0.703242 |
| LCA | 12 | 12 | 1 | 0.966100 |
| Alstrom | 9 | 9 | 1 | 0.996954 |
| Mitochondrial | 10 | 10 | 1 | 1.000000 |

Interpretation:
- The current baseline strongly reproduces expected top-module assignments on the bundled classic benchmark.
- Non-perfect probabilities (for example Choroideremia and Isolated RP) still ranked correctly at position 1, indicating discriminative but not absolute certainty.

---

## 5) Inputs and Data Provenance

All primary files are expected under `Input/`.

Core files:
- `module_all_HPO_background_comparison_20260413_1019.xlsx`
- `gene_classification_20260412_1524.csv`
- `hp.obo`
- `module_phenotypic_signatures_FDR_corrected_20260412_1524.csv`

Reference provenance (current implementation):
- HPO release: 2026-04-13
- IRD genes in scope: 442
- Disease modules: 17
- Module definitions: MPV network clustering
- Prevalence unit in engine: fraction [0, 1]

---

## 6) Installation and Quick Start

### Prerequisites
- Python >= 3.10
- pip

### Install

```bash
pip install -r requirements.txt
```

### Run Web App

```bash
streamlit run app.py
```

### Minimal API Example

```python
from clinical_support import ClinicalSupportEngine

engine = ClinicalSupportEngine(eager=True)

result = engine.query(
    observed=["HP:0000510", "HP:0000407"],
    excluded=["HP:0001513"],
)

print("Top module:", result.top_module.module_id)
print("Probability:", result.top_module.probability)
print("Confidence:", result.confidence)
print("Top gene:", result.candidate_genes[0].gene if result.candidate_genes else None)
```

---

## 7) Disease Modules (0-16)

| Module | Clinical Label |
|---|---|
| 0 | Macular Dystrophy / Broad RP |
| 1 | Syndromic Optic Atrophy (Mitochondrial) |
| 2 | Peroxisomal Syndromic |
| 3 | Transcriptional Regulation |
| 4 | IFT / Ciliary Dynein (Skeletal Ciliopathy) |
| 5 | Melanosome / OCA (Albinism) |
| 6 | CSNB / Phototransduction |
| 7 | Uncharacterized Syndromic |
| 8 | Familial Exudative Vitreoretinopathy (EVR) / Extracellular Matrix |
| 9 | Bardet-Biedl Syndrome (BBS) / Ciliopathy |
| 10 | Optic Atrophy / Mitochondrial Metabolism |
| 11 | Joubert Syndrome / Ciliopathy |
| 12 | Leber Congenital Amaurosis (LCA) |
| 13 | Syndromic Optic Atrophy (minor) |
| 14 | Cone Dystrophy / CRD |
| 15 | Color Vision Defects (CVD) / Cone Phototransduction |
| 16 | Usher Syndrome |

---

## 8) Project Structure

```text
MPV_phenotype_engine/
├── Input/
│   ├── module_all_HPO_background_comparison_20260413_1019.xlsx
│   ├── gene_classification_20260412_1524.csv
│   ├── module_phenotypic_signatures_FDR_corrected_20260412_1524.csv
│   ├── hp.obo
│   ├── IRD_vs_Non-IRD/
│   ├── modules_RetiGene_Comparison/
│   └── comparative_signatures_all_IRD/
├── app.py
├── clinical_support.py
├── data_loader.py
├── gene_ranker.py
├── hpo_traversal.py
├── output_models.py
├── prediction_engine.py
├── scoring_engine.py
├── session_manager.py
├── validation/
│   ├── classic_cases.py
│   └── verify_framework.py
└── requirements.txt
```

---

## 9) Known Extension Points

- Patient-cohort calibration layer (posterior calibration)
- NPP integration in gene ranking (`npp_score` currently reserved)
- Module prior customization (currently uniform)
- Ancestor weighting calibration using patient-level outcomes


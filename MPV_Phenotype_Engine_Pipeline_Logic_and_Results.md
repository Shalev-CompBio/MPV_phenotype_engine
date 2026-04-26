# MPV Phenotype Engine

A probabilistic clinical decision-support system for Inherited Retinal Disease (IRD) diagnosis.

This engine accepts partial phenotype evidence (HPO terms observed and/or excluded) and returns:
- Posterior probability over 17 IRD disease modules
- Ranked candidate genes within the top module
- Active-learning next questions (information gain)
- Predicted phenotypes for workup and prognosis
- HPO hierarchy-aware likely next manifestations

---

## 1) Current Status

This document is an updated technical README aligned to the current implementation in:
- `data_loader.py`
- `hpo_traversal.py`
- `scoring_engine.py`
- `gene_ranker.py`
- `prediction_engine.py`
- `clinical_support.py`
- `session_manager.py`
- `app.py`

It covers:
- Complete logic and pipeline steps
- Precise description of produced results
- Current validation outcomes from the codebase

> **Note**: `app.py` contains two generations of UI functions. The first set (lines ~2001–2831) is a legacy implementation retained for reference. The second set (lines ~2833–3590) is the **active redesigned UI** actually dispatched at runtime. All descriptions below refer to the active redesigned layer.

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

`DataLoader` builds the following indexes at startup:

| Index | Type | Description |
|---|---|---|
| `(hpo_id, module_id) → prevalence_fraction` | `dict[tuple, float]` | Fraction [0,1] of genes in module with phenotype |
| `hpo_id → background_prevalence` | `dict[str, float]` | Mean prevalence across all 17 modules |
| `hpo_id → phenotype_name` | `dict[str, str]` | Human-readable term label |
| `name_lower → hpo_id` | `dict[str, str]` | Free-text lookup |
| `module_id → genes[]` | `dict[int, list]` | All genes assigned to each module |
| `gene → {module_id, stability_score, classification}` | `dict[str, dict]` | Per-gene metadata |
| `gene → set(hpo_ids)` | `dict[str, set]` | Inverted annotation index from module sheets |
| `module_id → significant_signature_terms[]` | `dict[int, list]` | FDR-significant phenotypic signatures per module |
| `hpo_id → set(module_ids)` | `dict[str, set]` | Which modules a term is a signature for |

Prevalence lookup falls back in this order:
1. Exact `(hpo_id, module_id)` entry
2. Module-agnostic background mean for the HPO term
3. Global floor `0.001` (0.1%)

### 2.3 Ontology Traversal Layer

`HPOTraversal` provides:
- **Upward traversal**: ancestors with distance-decaying weights
- **Downward traversal**: children filtered to the IRD annotation space (`ird_terms`)
- **Free-text resolution**: term name → canonical HPO ID

Ancestor weight schedule:

| Distance | Weight |
|---|---|
| 1 | 0.80 |
| 2 | 0.60 |
| 3 | 0.40 |
| ≥ 4 | 0.20 (floor) |

### 2.4 Probabilistic Scoring Layer

`ScoringEngine.score_modules(observed, excluded)` executes in three steps:

**Step 1 — Build weighted observed map:**
- Direct observed term weight = 1.0
- Add ancestor terms with their decay weights
- If a term appears via multiple paths, keep the maximum weight (not the sum)

**Step 2 — Compute module log-likelihood:**

$$
\log L(m) = \sum_{h \in O_w} w_h \log(\tilde{p}_{h,m}) + \sum_{h \in E} \log(1 - \tilde{p}_{h,m})
$$

where:
- $O_w$: weighted observed map (direct observed terms + expanded ancestors)
- $E$: excluded terms (direct only — ancestors are not expanded for exclusions)
- $\tilde{p}_{h,m}$: prevalence of term $h$ in module $m$, clipped to $[0.001,\ 0.999]$

**Step 3 — Normalize:**
Apply numerically stable softmax over all 17 log-likelihoods to obtain the posterior distribution.

### 2.5 Confidence Layer

Confidence measures how concentrated the posterior is:

$$
\text{confidence} = 1 - \frac{H(p)}{\log_2(17)}
$$

$$
H(p) = -\sum_i p_i \log_2 p_i
$$

Interpretation:
- **0.0** — maximal uncertainty: posterior is flat across all 17 modules
- **1.0** — maximal certainty: all probability mass on one module

Confidence is not an external calibration probability. It reflects the discriminative sharpness of the current evidence, not the absolute correctness of the top call.

### 2.6 Gene Ranking Layer — Soft Module-Aware Gene Scoring (SMA-GS)

`GeneRanker.rank_genes(module_id, observed)` scores every gene in the top module against the patient's observed HPO terms using the SMA-GS formula.

**Motivation:** Gene-phenotype annotations are incomplete. A core gene that co-clusters with others known to cause a phenotype probably also causes it, even if not yet recorded in the literature. SMA-GS provides calibrated soft credit for such plausible but unannotated links, gated by how representative the gene is of its module.

#### Coherence Gate ψ(g)

The coherence gate encodes how representative gene $g$ is of its disease module, based on its cluster stability classification:

| Classification | ψ(g) |
|---|---|
| `core` | 1.0 |
| `peripheral` | 0.5 |
| `unstable` | 0.0 |

Unstable genes receive no leakage credit regardless of γ. Core genes receive full leakage credit. Peripheral genes receive half.

#### Affinity Function A(g, h)

For each observed HPO term $h$ and gene $g$:

$$
A(g, h) = \underbrace{1_{h \in G_g}}_{\text{exact match}} + \underbrace{(1 - 1_{h \in G_g}) \cdot \gamma \cdot \psi(g)}_{\text{leakage}}
$$

where:
- $G_g$: set of HPO IDs annotated to gene $g$ in the module sheets
- $\gamma \in [0, 1]$: global leakage parameter, user-configurable via sidebar slider (default 0.3)
- $\psi(g)$: coherence gate defined above

The two components are mutually exclusive:
- If $h \in G_g$ (annotated): affinity = 1.0, leakage = 0
- If $h \notin G_g$ (not annotated): exact match = 0, leakage = $\gamma \cdot \psi(g)$

#### Final Score

$$
\text{score}(g) = \frac{\displaystyle\sum_{h \in O} A(g,h) \cdot p(h,m)}{\max\!\left(\displaystyle\sum_{h \in O} p(h,m),\ 10^{-9}\right)}
$$

where $p(h,m)$ is the prevalence of term $h$ in the top module $m$. Terms with $p(h,m) = 0$ are skipped (they contribute nothing to either the numerator or denominator).

The denominator is the total prevalence mass of the query in the module — normalizing so that a gene annotated to all observed terms scores ≤ 1.0. The floor $10^{-9}$ prevents division by zero when no observed term has any prevalence in the module.

#### Behavior at Extreme γ Values

| γ | Behavior |
|---|---|
| 0.0 | Strict binary matching. Only annotated gene-phenotype pairs score. Unstable and peripheral genes score identically to core if their annotations match. |
| 0.3 | Default. Core genes receive modest soft credit (~30%) for unannotated module symptoms. Peripheral genes receive ~15%. Unstable genes receive none. |
| 1.0 | Maximum leakage. Core genes receive full module-prevalence credit for every observed symptom, regardless of annotation status. |

Note: γ=0 does not reproduce the pre-SMA-GS system exactly. The previous implementation applied an additive stability modifier (`stability_score × 0.2 × direction`) on top of binary overlap. SMA-GS removes this modifier entirely; the coherence gate now operates only through the leakage term, not as an additive score component.

#### Outputs Per Gene

Each `GeneCandidate` contains:

| Field | Content |
|---|---|
| `score` | Final SMA-GS score, rounded to 6 decimal places |
| `stability` | Classification string: `"core"`, `"peripheral"`, or `"unstable"` |
| `supporting_phenotypes` | Names of observed terms the gene is **directly annotated** to (exact matches only; leak terms excluded) |
| `score_breakdown` | `list[tuple[str, float]]` — (term name, total contribution) for all terms with positive affinity, sorted descending. Includes both direct and leak terms. |
| `leak_breakdown` | `list[tuple[str, float]]` — (term name, imputed leakage contribution) for leak-only terms. Empty when γ=0 or gene is unstable. |
| `stability_breakdown` | `tuple[str, float, float]` — (classification, ψ, total_gate_contribution). `ψ` is the coherence gate value; `total_gate_contribution` is the sum of all leak contributions. |
| `npp_score` | `None` (reserved for future protein network prior integration) |

### 2.7 Prediction + Active Learning Layer

`PredictionEngine` returns three phenotype bundles and a ranked question list:

**Recommended workup:**
- Terms with module prevalence ≥ 50%
- Excluding already-observed terms and their ancestors
- Generic terms with background prevalence ≥ 80% across all modules are filtered out

**Prognostic risk:**
- Terms with module prevalence in [15%, 50%)

**Likely next manifestations:**
- Depth-1 children of observed terms in the HPO graph
- Must be in the IRD annotation space and have module prevalence > 0
- Deduplicated against workup and risk lists

**Next questions (active learning):**
- Candidate pool: terms with max prevalence ≥ 5% in any module, capped at 300
- Already-asked terms are excluded

Information gain per candidate $q$:

$$
p_{\text{yes}} = \sum_m p(m) \cdot P(q \mid m), \quad p_{\text{no}} = 1 - p_{\text{yes}}
$$

$$
IG(q) = H(p) - \left[ p_{\text{yes}} H(p \mid q{=}\text{yes}) + p_{\text{no}} H(p \mid q{=}\text{no}) \right]
$$

Entropy is computed in nats. Top-5 questions by IG are returned.

Qualitative IG labels used in the UI:

| Label | Threshold |
|---|---|
| High diagnostic value | ≥ 0.8 nats |
| Moderate diagnostic value | 0.3 – 0.8 nats |
| Low diagnostic value | < 0.3 nats |

### 2.8 Orchestration Layer

`ClinicalSupportEngine` is the public entry point. It instantiates and holds all sub-modules, and assembles the full `QueryResult` in a fixed pipeline order:

1. Score all 17 modules (`ScoringEngine`)
2. Compute confidence (entropy normalization)
3. Identify top module (argmax of posterior)
4. Rank genes within top module (`GeneRanker`)
5. Predict workup / risk / next manifestations (`PredictionEngine`)
6. Suggest top-5 next questions by information gain (`PredictionEngine`)

The γ parameter is passed to `ClinicalSupportEngine` at initialization and forwarded to `GeneRanker`. Changing γ via the UI slider creates a new engine instance (via `@st.cache_resource` keyed on the slider value).

Public methods:
- `query(observed, excluded)` — full phenotype batch query
- `query_gene(gene_symbol)` — gene-first query using the gene's annotated HPO terms as observed
- `new_session()` — returns a stateful `Session` object for interactive Q&A
- `browse_module(module_id)` — returns structured display data for the Module Browser view

### 2.9 UI/Delivery Layer

`app.py` provides four navigation modes dispatched from the sidebar:
- **Phenotype Query** — batch HPO phenotype input + gene-first query
- **Interactive Session** — one-question-at-a-time active learning loop
- **Module Browser** — per-module HPO profile, signatures, genes, and system architecture plots
- **Comparative Analytics** — inter-module enrichment and global IRD vs. universe context

**Engine Parameters sidebar panel** (rendered before engine initialization):
- **Module leakage (γ)** slider — range [0.0, 1.0], default 0.3. Controls the SMA-GS leakage weight. Moving the slider immediately re-instantiates the engine with the new γ value (cached per unique value, so returning to a prior value is free).

Additional UI-specific behaviors:
- Sticky topbar showing HPO version, gene count, and engine-ready status
- Clinical Case Library (5 validated synthetic cases) rendered as buttons above the query form
- Real patient-derived cases loaded from `Input/real_clinical_cases_signal_only_13.04.26.csv` and rendered in a collapsible expander
- Add-to-query reactive rerun from workup / risk / next manifestation lists
- Session history with per-entry **Undo** button and confidence progression tracker
- Expandable gene rows in the Candidate Genes tab showing phenotype contribution bars, coherence gate (ψ), and module-extrapolated contributions panel for leak terms
- Downloadable CSV for candidate genes and optional PDF summary (requires `reportlab`)
- Module signature highlighting (⭐) in phenotype output names and gene table
- System architecture PNGs for each module in the Module Browser with a download button
- Dark-mode style overrides (CSS injected when `dark_mode` session flag is set; currently forced to `False`)

---

## 3) Results: What the Engine Produces

For each run, the engine returns a complete `QueryResult` containing:

1. Top module and full posterior distribution (all 17 modules)
2. Confidence score (entropy-normalized)
3. Ranked candidate genes (within the top module)
4. Phenotype prediction bundles: recommended workup, prognostic risk, likely next manifestations
5. Active-learning question suggestions (top 5 by information gain)

### 3.1 Result Rendering (UI)

The active `_render_result()` function renders results across three tabs:

**Overview tab:**
- Hero card: top module + posterior probability
- SVG confidence gauge
- Top-3 module summary cards
- Full 17-module posterior bar chart with prior line at 1/17
- Top-5 next questions as inline information-gain bar cards

**Candidate Genes tab:**
- Expandable row table sorted by score, cluster confidence, % phenotype match, or matching HPO count
- Per-gene expandable panel shows:
  - Phenotype contribution bars (all terms with positive affinity, direct + leak)
  - Coherence gate display: stability classification and ψ value
  - Module-aware credit line showing the total leakage contribution
  - Module-extrapolated contributions sub-panel listing each leak term individually (only visible when γ > 0 and gene is not unstable)
  - When `score_breakdown` is empty: message "No phenotype overlap with query — score is 0 for this gene"
- CSV and PDF download buttons

**Workup and prognosis tab:**
- Three-column layout: Recommended Workup (≥ 50%), Prognostic Risk (15–50%), Likely Next Manifestations (ontology children)
- Each column has "+ Add" buttons that insert the term into the query and trigger a rerun

### 3.2 Result Semantics

| Output | Interpretation |
|---|---|
| Module probability | Bayesian posterior conditioned on observed/excluded evidence under Naive Bayes assumptions |
| Confidence | Sharpness of the posterior distribution — how concentrated evidence is on one module. Not an absolute correctness probability. |
| Gene score | Prevalence-weighted affinity between gene and patient phenotype, gated by cluster membership coherence. Ranges [0, 1+] theoretically; in practice bounded by annotation density. |
| IG score | Expected reduction in posterior entropy from asking the next question. Computed in nats. |

### 3.3 Output Model

Primary dataclasses from `output_models.py`:

**`ModuleMatch`**
- `module_id: int`
- `module_label: str | None` — reserved for manual clinical annotation
- `probability: float`
- `supporting_phenotypes: list[str]` — names of observed terms with prevalence > 0 in this module

**`GeneCandidate`**
- `gene: str`
- `score: float` — SMA-GS final score
- `stability: str` — `"core"` | `"peripheral"` | `"unstable"`
- `supporting_phenotypes: list[str]` — directly annotated matching terms only (no leak terms)
- `npp_score: float | None` — reserved; currently `None`
- `score_breakdown: list[tuple[str, float]]` — (term name, total contribution) for all terms with positive affinity, sorted descending
- `leak_breakdown: list[tuple[str, float]]` — (term name, leakage contribution) for imputed terms only; empty when γ=0 or gene is unstable
- `stability_breakdown: tuple[str, float, float] | None` — (classification, ψ, total_gate_contribution)

**`HPOTerm`**
- `hpo_id: str`, `term_name: str`, `prevalence: float | None`

**`PhenotypePrediction`**
- `recommended_workup: list[HPOTerm]`
- `prognostic_risk: list[HPOTerm]`
- `likely_next_manifestations: list[HPOTerm]`

**`SuggestedQuestion`**
- `hpo_id: str`, `term_name: str`, `information_gain: float`

**`QueryResult`**
- `top_module: ModuleMatch`
- `all_modules: list[ModuleMatch]` — all 17, sorted by probability descending
- `candidate_genes: list[GeneCandidate]` — sorted by score descending
- `phenotype_predictions: PhenotypePrediction`
- `next_question: SuggestedQuestion` — top-1 question (also the first entry of `next_questions`)
- `next_questions: list[SuggestedQuestion]` — top-5 questions
- `confidence: float`

---

## 4) Validation Results

Validation was executed using the classic case suite defined in `validation/classic_cases.py`.

**Summary: 9/9 cases — top-ranked module matched expected module in all cases.**

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

Non-perfect probabilities (e.g., Choroideremia at 0.70) still rank correctly at position 1, reflecting genuine phenotypic overlap between modules rather than scoring failure. The module scoring layer is unchanged by SMA-GS.

---

## 5) Inputs and Data Provenance

All primary files are expected under `Input/`.

| File | Role |
|---|---|
| `module_all_HPO_background_comparison_20260413_1019.xlsx` | 17-sheet prevalence matrix |
| `gene_classification_20260412_1524.csv` | Gene → module, stability score, classification |
| `hp.obo` | HPO ontology graph |
| `module_phenotypic_signatures_FDR_corrected_20260412_1524.csv` | FDR-significant module signatures |

Reference provenance (current implementation):
- HPO release: 2026-04-13
- IRD genes in scope: 442
- Disease modules: 17
- Module definitions: MPV network clustering
- Prevalence unit in engine: fraction [0, 1] (xlsx values are %, divided by 100)

---

## 6) Installation and Quick Start

### Prerequisites
- Python ≥ 3.10

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

engine = ClinicalSupportEngine(eager=True, gamma=0.3)

result = engine.query(
    observed=["HP:0000510", "HP:0000407"],
    excluded=["HP:0001513"],
)

print("Top module:", result.top_module.module_id)
print("Probability:", result.top_module.probability)
print("Confidence:", result.confidence)

top_gene = result.candidate_genes[0]
print("Top gene:", top_gene.gene)
print("Score:", top_gene.score)
print("Stability:", top_gene.stability, f"(ψ={top_gene.stability_breakdown[1]:.2f})")
print("Direct matches:", top_gene.supporting_phenotypes)
print("Leak terms:", [name for name, _ in top_gene.leak_breakdown])
```

---

## 7) Disease Modules (0–16)

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
│   ├── real_clinical_cases_signal_only_13.04.26.csv
│   ├── IRD_vs_Non-IRD/
│   │   ├── IRD_vs_Non-IRD.csv
│   │   └── signature_barplot_ird_universe_LIST.png
│   ├── modules_RetiGene_Comparison/
│   │   ├── enrichment_results_260412_1551.csv
│   │   ├── enrichment_dotplot_260412_1551.png
│   │   ├── dual_coherence_scatter_20260421_1807.png
│   │   ├── hpo_coherence_stripplot_260423_1021.png
│   │   └── npp_coherence_stripplot_260421_1625.png
│   ├── comparative_signatures_all_IRD/
│   │   └── comparative_signatures_20260413_0959.pdf
│   └── single_system_plots/
│       ├── system_plot_module_0.png … system_plot_module_16.png
│       └── system_plot_module_All.png
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

| Extension | Status | Notes |
|---|---|---|
| NPP integration in gene ranking | Reserved | `npp_score` field exists in `GeneCandidate`; formula slot is documented in `gene_ranker.py` |
| Patient-cohort posterior calibration | Not implemented | Would convert posterior → calibrated probability |
| Module prior customization | Not implemented | Currently uniform prior (1/17 per module) |
| Ancestor weighting calibration | Not implemented | Decay schedule currently hardcoded |
| γ calibration via MRR | Manually tuned | Default 0.3 chosen by MRR sweep on 13 real clinical cases; can be re-tuned as cohort grows |

---

## 10) Comparative Analytics — Module Coherence Landscape

The **Comparative Analytics** mode includes a dedicated **Module Coherence Landscape** section:

| Visualisation | File | Description |
|---|---|---|
| Dual Coherence Scatter | `dual_coherence_scatter_20260421_1807.png` | Each module positioned by median ΔHPO (x) vs median ΔNPP (y). Upper-right quadrant = coherent in both phenotype and protein-network spaces. |
| HPO Coherence Stripplot (ΔHPO) | `hpo_coherence_stripplot_260423_1021.png` | Per-gene cosine similarity to cluster HPO centroid, ranked by module median. Marker shape encodes stability classification. |
| NPP Coherence Stripplot (ΔNPP) | `npp_coherence_stripplot_260421_1625.png` | Per-gene cosine similarity to cluster protein-interaction centroid. Marker fill encodes permutation p-value tier. |

All three plots have associated download buttons in the UI.

The analytics header also renders four hardcoded summary statistics cards:
- **Validation cases**: 9/9
- **Avg Recall**: 86.2%
- **Avg Precision**: 94.1%
- **Novel candidates**: +15

---

## 11) app.py UI Architecture Notes

### Dual-Definition Pattern

`app.py` contains two full sets of mode functions. Python resolves duplicate function names to the **last definition**, so the v2 functions are always dispatched at runtime. The v1 functions are retained for reference only.

| Function | Approximate Lines | Status |
|---|---|---|
| `_render_result()` v1 | ~1596–1994 | Legacy — uses `altair` bar chart |
| `_query_mode()` v1 | ~2001–2123 | Legacy |
| `_session_mode()` v1 | ~2129–2327 | Legacy |
| `_analytics_mode()` v1 | ~2329–2621 | Legacy |
| `_browser_mode()` v1 | ~2627–2831 | Legacy |
| `_render_result()` v2 | ~2833–2986 | **Active** — uses custom HTML chart |
| `_query_mode()` v2 | ~2989–3088 | **Active** |
| `_session_mode()` v2 | ~3091–3272 | **Active** |
| `_browser_mode()` v2 | ~3274–3440 | **Active** — 4 tabs |
| `_analytics_mode()` v2 | ~3442–3590 | **Active** — includes coherence landscape |

Line numbers are approximate and shift as the file is edited.

### Engine Initialization and γ Caching

```python
@st.cache_resource
def _load_engine(gamma: float = 0.3) -> ClinicalSupportEngine:
    return ClinicalSupportEngine(eager=True, gamma=gamma)
```

The γ slider is rendered in the sidebar before `_load_engine` is called. `@st.cache_resource` keys on the argument, so each unique γ value maps to one cached engine instance. Switching from 0.3 → 0.4 → 0.3 reuses the cached 0.3 instance on the return trip.

### Module Browser Tabs (v2)

Four tabs:
1. **HPO Profile** — top terms as inline bar rows (top 25 by module prevalence)
2. **Signatures** — FDR-significant enrichments as styled cards with odds ratio and q-value
3. **Gene List** — stability-ranked gene cards with cluster confidence badges
4. **System Architecture** — per-module PNG network plot + all-module overview

### Session Mode (v2)

Two-column layout:
- **Left column**: Current question card with IG bar + answer buttons (Yes / No / Skip) + live "Current Diagnosis" result below
- **Right column**: Session history as color-coded answer cards with per-entry Undo button + confidence progression tracker

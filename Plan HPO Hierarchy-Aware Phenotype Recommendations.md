Plan: HPO Hierarchy-Aware Phenotype Recommendations
Context
The Recommended Workup and Prognostic Risk lists are currently built by a simple prevalence threshold + MAX_TERMS cap. Two clinically meaningful improvements use the existing HPO ontology (already loaded via hpo_traversal.py):

Ancestor suppression: If a patient already has "Macular degeneration" (specific), listing its ontology parent "Abnormality of the macula" (general) in the workup adds no clinical value. Filter ALL ancestors of observed terms from both lists.

Likely Next Manifestations: Direct children of observed HPO terms represent the most probable next step in disease progression. They are currently at risk of being crowded out by high-prevalence parent terms hitting the MAX_TERMS cap. Surface them in a dedicated output block, with no minimum prevalence threshold (relevance comes from the ontology relationship, not raw frequency).

Files to Modify
File	Change
output_models.py	Add likely_next_manifestations field to PhenotypePrediction
prediction_engine.py	Build ancestor set, add filter, build progression candidates
app.py	Add third column + CSS for new section; update PDF export
hpo_traversal.py and clinical_support.py require no changes.

Step 1 — output_models.py (line 30–32)
Add one field with a default to PhenotypePrediction:

@dataclass
class PhenotypePrediction:
    recommended_workup: list[HPOTerm]
    prognostic_risk: list[HPOTerm]
    likely_next_manifestations: list[HPOTerm] = field(default_factory=list)
field is already imported on line 2. The default ensures backward compatibility for any direct construction of this dataclass.

Step 2 — prediction_engine.py
2a. New constant (alongside MAX_TERMS at line 48)
MAX_PROGRESSION_TERMS = 10  # cap for likely_next_manifestations
2b. predict_phenotypes() body changes
After line 86 (observed_set = set(observed)), insert:

# All ancestors of observed terms — these are ontologically subsumed
# by the observed findings and add no clinical specificity.
ancestors_of_observed: set[str] = set()
for obs_id in observed:
    for anc_id, _ in self._ht.get_ancestors(obs_id):
        ancestors_of_observed.add(anc_id)
In the classification loop (lines 109–119), add one guard after if hpo_id in seen:

for hpo_id, prev in module_terms:
    if hpo_id in seen:
        continue
    if hpo_id in ancestors_of_observed:   # NEW
        continue
    if prev >= WORKUP_THRESHOLD:
        ...
    elif prev >= RISK_THRESHOLD:
        ...
After line 123 (after risk = self._expand_with_children(...)), before the return:

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
Update return statement:

return PhenotypePrediction(
    recommended_workup=workup,
    prognostic_risk=risk,
    likely_next_manifestations=likely_next,
)
Edge cases handled:

observed = [] → both ancestor set and next candidates are empty (no-op)
Unknown HPO ID → get_ancestors() returns [] (safe, per hpo_traversal.py:76)
Diamond inheritance → seen_next deduplicates children shared by multiple observed terms
get_children() already filters to IRD-space terms; child_prev > 0 is an extra safety net
Step 3 — app.py
3a. New CSS (after .risk-item block, ~line 143)
.next-item {
    border-left: 4px solid #2a9d8f;
    background: #eaf7f5;
    border-radius: 0 6px 6px 0;
    padding: 5px 10px;
    margin: 3px 0;
    font-size: 0.9rem;
}
In the dark mode override block:

.next-item { background: #0f2420 !important; border-left-color: #2a9d8f !important; }
3b. Tab layout (line 647)
Change two columns to three:

col_workup, col_risk, col_next = st.columns(3)
Add a with col_next: block after the existing with col_risk: block (after line 715), before the footer st.markdown:

with col_next:
    st.subheader("Likely Next Manifestations")
    st.markdown(
        '<p class="explain">Direct ontology children of observed phenotypes '
        "present in this module — the most probable next step in disease "
        "progression, surfaced regardless of prevalence threshold.</p>",
        unsafe_allow_html=True,
    )
    nxt = result.phenotype_predictions.likely_next_manifestations
    if nxt:
        for i, t in enumerate(nxt):
            pct = f" — {t.prevalence * 100:.0f}%" if t.prevalence else ""
            div = (
                f'<div class="next-item">🔭 <b>{t.term_name}</b> '
                f'<code>{t.hpo_id}</code>{pct}</div>'
            )
            if workup_add_to_query:
                row_l, row_r = st.columns([5, 1])
                with row_l:
                    st.markdown(div, unsafe_allow_html=True)
                with row_r:
                    if st.button(
                        "Add",
                        key=f"nx_add_{t.hpo_id}_{i}",
                        help="Add to observed phenotypes",
                    ):
                        if _add_hpo_to_query_observed(t.hpo_id):
                            st.rerun()
                        else:
                            st.toast("This term is not available in the phenotype list.", icon="⚠️")
            else:
                st.markdown(div, unsafe_allow_html=True)
    else:
        st.markdown("*No progression phenotypes identified from observed terms.*")
3c. PDF export (after line 416, before the Spacer at line 418)
# Likely Next Manifestations
story.append(Spacer(1, 0.10 * inch))
story.append(Paragraph("Likely Next Manifestations", head_style))
nxt = result.phenotype_predictions.likely_next_manifestations
if nxt:
    for t in nxt:
        pct = f" — {t.prevalence * 100:.0f}%" if t.prevalence else ""
        story.append(Paragraph(
            f"🔭 <b>{t.term_name}</b> ({t.hpo_id}){pct}", body_style
        ))
else:
    story.append(Paragraph("No likely next manifestations identified.", body_style))
Verification
Ancestor suppression: Observe a leaf HPO term → confirm its direct parent and grandparent are absent from workup and risk output.
Empty observed: predict_phenotypes(module_id=0, observed=[]) → same output as before, likely_next_manifestations == [], no exceptions.
Children appear in likely_next: Observe a term with known IRD-space children → confirm those children appear in likely_next_manifestations and not duplicated in prognostic_risk.
No duplication with workup: If a child of an observed term has prevalence ≥50%, it appears in recommended_workup, not in likely_next_manifestations.
Cap enforcement: likely_next_manifestations has at most 10 items, sorted by prevalence descending.
PDF: Builds without error; "Likely Nex
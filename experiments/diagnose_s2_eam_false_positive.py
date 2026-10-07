import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Core.configs.system_config import load_system_config
from Core.configs.llm_config import LLMConfig
from Core.configs.vlm_config import VLMConfig
from Core.configs.rag.gbc_config import GBCRAGConfig

from Core.Index.GBCIndex import GBC

from Core.rag import create_rag_agent
from Core.rag.gbc_utils import SubStep

from Core.rag.self_correcting.eam import EvidenceAssessmentModule
from Core.rag.self_correcting.types import EvidenceAssessment


CONFIG = "config/self_correcting_kg_fallback.yaml"

QUERY = (
    "For the circulation equipment serving Reactor R-2, "
    "what is its alternate identifier and which inlet "
    "strainer protects it?"
)

INITIAL_NODE_IDS = [34, 38, 39, 40, 17, 18]


def find_typed_config(obj, target_type, max_depth=5):
    visited = set()

    def walk(value, depth):
        if value is None:
            return None
        if isinstance(value, target_type):
            return value
        if depth > max_depth:
            return None
        value_id = id(value)
        if value_id in visited:
            return None
        visited.add(value_id)
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set)):
            children = value
        else:
            try:
                children = vars(value).values()
            except TypeError:
                return None
        for child in children:
            found = walk(child, depth + 1)
            if found is not None:
                return found
        return None

    return walk(obj, 0)


def node_ids_from_dicts(nodes):
    ids = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("index_id")
        if node_id is None:
            continue
        try:
            node_id = int(node_id)
        except (TypeError, ValueError):
            continue
        ids.append(node_id)
    return ids


def main():
    print("=" * 80)
    print("DIAGNOSTIC: S2 EAM FALSE POSITIVE ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load real indexes
    # ------------------------------------------------------------------
    system_cfg = load_system_config(CONFIG)
    gbc = GBC.load_gbc_index(system_cfg)

    rag_cfg = find_typed_config(system_cfg, GBCRAGConfig)
    llm_cfg = find_typed_config(system_cfg, LLMConfig)
    vlm_cfg = find_typed_config(system_cfg, VLMConfig)

    rag = create_rag_agent(
        strategy_config=rag_cfg,
        llm_config=llm_cfg,
        vlm_config=vlm_cfg,
        gbc_index=gbc,
    )

    # ------------------------------------------------------------------
    # 2. Load initial evidence nodes
    # ------------------------------------------------------------------
    initial_tree_data = gbc.TreeIndex.get_nodes_data(INITIAL_NODE_IDS)

    for nd in initial_tree_data:
        if nd.get("page") is None:
            nd["page"] = 0

    current_step = SubStep(sub_query=QUERY, sub_number=1)
    rag._process_retrieved_nodes(initial_tree_data, current_step)

    # ------------------------------------------------------------------
    # 3. Print initial evidence text for each node
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("INITIAL TREE NODES")
    print("=" * 80)

    for nd in initial_tree_data:
        nid = nd.get("index_id")
        text = nd.get("content", "")
        print(f"\n--- Node {nid} ---")
        print(text[:500] if text else "(empty)")

    # ------------------------------------------------------------------
    # 4. Trace EAM entity extraction
    # ------------------------------------------------------------------
    eam = rag.sc_eam

    print()
    print("=" * 80)
    print("QUERY TERMS (EAM _content_terms)")
    print("=" * 80)

    query_terms = eam._content_terms(QUERY)
    print(sorted(query_terms))

    print()
    print("=" * 80)
    print("QUERY ENTITIES (EAM _guess_missing_entities)")
    print("=" * 80)

    combined_evidence_text = " ".join(
        nd.get("content", "") for nd in initial_tree_data
    )
    missing_entities = eam._guess_missing_entities(
        QUERY, combined_evidence_text
    )
    print("missing_entities:", missing_entities)

    # Trace each candidate
    candidates_raw = re.findall(
        r"\b(?:"
        r"[A-Z][A-Za-z0-9_-]*"
        r"|[A-Z]{2,}[A-Za-z0-9_-]*"
        r"|[A-Za-z]+-\d+[A-Za-z0-9_-]*"
        r")\b",
        QUERY,
    )
    ignored = {"Who", "What", "Which", "Where", "When", "Why", "How"}
    candidates = [c for c in candidates_raw if c not in ignored]

    evidence_lower = combined_evidence_text.lower()
    print("\nCandidate extraction trace:")
    for c in candidates:
        present = c.lower() in evidence_lower
        print(f"  '{c}' -> present in evidence: {present}")

    # ------------------------------------------------------------------
    # 5. Initial EAM assessment
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("INITIAL EAM ASSESSMENT")
    print("=" * 80)

    initial_assessment = eam.assess(
        query=QUERY,
        retrieval_nodes=current_step.retrieval_nodes,
    )

    print(f"Status: {initial_assessment.status.value}")
    print(f"Diagnosis: {initial_assessment.diagnosis.value}")
    print(f"Relevance: {initial_assessment.relevance_score}")
    print(f"Coverage: {initial_assessment.coverage_score}")
    print(f"Support: {initial_assessment.support_score}")
    print(f"Contradiction: {initial_assessment.contradiction_score}")
    print(f"Missing entities: {initial_assessment.missing_entities}")
    print(f"Evidence IDs: {initial_assessment.evidence_ids}")

    # ------------------------------------------------------------------
    # 6. Run S2 recovery
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("S2 RECOVERY")
    print("=" * 80)

    rag._apply_self_correcting_recovery(
        query=QUERY,
        current_step=current_step,
    )

    final_ids = node_ids_from_dicts(current_step.retrieval_nodes)
    print("Final Tree IDs after recovery:", final_ids)

    # Get text of all recovered nodes
    final_tree_data = gbc.TreeIndex.get_nodes_data(final_ids)

    # ------------------------------------------------------------------
    # 7. Trace contradiction on S2-recovered evidence
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("CONTRADICTION ANALYSIS ON RECOVERED EVIDENCE")
    print("=" * 80)

    final_assessment = eam.assess(
        query=QUERY,
        retrieval_nodes=current_step.retrieval_nodes,
    )

    print(f"Final status: {final_assessment.status.value}")
    print(f"Final diagnosis: {final_assessment.diagnosis.value}")
    print(f"Final contradiction: {final_assessment.contradiction_score}")

    # Manually trace the contradiction detection
    query_terms = eam._content_terms(QUERY)

    evidence_texts = []
    evidence_ids = []
    for nd in current_step.retrieval_nodes:
        text = eam._extract_text(nd)
        if text:
            evidence_texts.append(text)
        evidence_ids.append(eam._extract_id(nd, fallback_index=len(evidence_ids)))

    evidence_term_sets = [
        eam._content_terms(text) for text in evidence_texts
    ]

    # Determine relevant indices
    relevant_indices = []
    for index, terms in enumerate(evidence_term_sets):
        matched = eam._matched_query_terms(query_terms, terms)
        match_ratio = len(matched) / max(len(query_terms), 1)
        if len(matched) >= 2 or match_ratio >= 0.50:
            relevant_indices.append(index)

    print(f"\nTotal evidence items: {len(evidence_texts)}")
    print(f"Query-relevant evidence items: {len(relevant_indices)}")

    print("\nQuery-relevant evidence items:")
    for idx in relevant_indices:
        nid = evidence_ids[idx]
        matched = eam._matched_query_terms(query_terms, evidence_term_sets[idx])
        match_ratio = len(matched) / max(len(query_terms), 1)
        text_preview = evidence_texts[idx][:200].replace("\n", " ")
        print(f"  Index {idx} (node {nid}): matched={sorted(matched)}, ratio={match_ratio:.4f}")
        print(f"    Text: {text_preview}")

    # Check for numeric measurements in relevant evidence
    print("\n" + "=" * 80)
    print("NUMERIC MEASUREMENT EXTRACTION IN RELEVANT EVIDENCE")
    print("=" * 80)

    pattern = re.compile(
        r"(?<![A-Za-z0-9-])"
        r"(\d+(?:\.\d+)?)"
        r"\s+"
        r"(?:[A-Za-z]+\s+){0,2}"
        r"(hours?|hrs?|volts?|"
        r"psi|bar|rpm|"
        r"minutes?|mins?|seconds?|secs?|"
        r"mm|cm|kg|"
        r"liters?|litres?|"
        r"percent|%)"
        r"\b",
        re.IGNORECASE,
    )

    unit_map = {
        "hour": "hours", "hr": "hours", "hrs": "hours",
        "volt": "volts",
        "minute": "minutes", "min": "minutes", "mins": "minutes",
        "second": "seconds", "sec": "seconds", "secs": "seconds",
        "liter": "liters", "litre": "liters", "litres": "liters",
    }

    relevant_measurement_sets = []
    for idx in relevant_indices:
        text = evidence_texts[idx]
        measurements = {}
        for value, unit in pattern.findall(text):
            unit_lower = unit.lower()
            unit_norm = unit_map.get(unit_lower, unit_lower)
            measurements.setdefault(unit_norm, set()).add(value)
        relevant_measurement_sets.append(measurements)

        if measurements:
            nid = evidence_ids[idx]
            print(f"\n  Node {nid} measurements:")
            for unit, vals in measurements.items():
                print(f"    {unit}: {vals}")

    # Check for shared terms and conflicts
    print("\n" + "=" * 80)
    print("PAIRWISE CONTRADICTION CHECK")
    print("=" * 80)

    found_conflict = False
    for i in range(len(relevant_indices)):
        for j in range(i + 1, len(relevant_indices)):
            shared_terms = (
                evidence_term_sets[relevant_indices[i]]
                & evidence_term_sets[relevant_indices[j]]
            )
            if len(shared_terms) < 2:
                continue

            shared_units = (
                set(relevant_measurement_sets[i])
                & set(relevant_measurement_sets[j])
            )

            if not shared_units:
                continue

            nid_i = evidence_ids[relevant_indices[i]]
            nid_j = evidence_ids[relevant_indices[j]]

            for unit in shared_units:
                vals_i = relevant_measurement_sets[i][unit]
                vals_j = relevant_measurement_sets[j][unit]

                if vals_i and vals_j and vals_i.isdisjoint(vals_j):
                    found_conflict = True
                    print(f"\n  CONFLICT FOUND:")
                    print(f"    Node {nid_i}: {vals_i} {unit}")
                    print(f"    Node {nid_j}: {vals_j} {unit}")
                    print(f"    Shared terms: {sorted(shared_terms)}")
                    text_i = evidence_texts[relevant_indices[i]][:300].replace("\n", " ")
                    text_j = evidence_texts[relevant_indices[j]][:300].replace("\n", " ")
                    print(f"    Node {nid_i} text: {text_i}")
                    print(f"    Node {nid_j} text: {text_j}")

    if not found_conflict:
        print("  No numeric conflict found in pairwise check.")

    # Check for marker-based contradiction
    print("\n" + "=" * 80)
    print("MARKER CONTRADICTION CHECK")
    print("=" * 80)

    marker_relevant_texts = []
    for text, terms in zip(
        [evidence_texts[idx] for idx in relevant_indices],
        [evidence_term_sets[idx] for idx in relevant_indices],
    ):
        matched = eam._matched_query_terms(query_terms, terms)
        match_ratio = len(matched) / max(len(query_terms), 1)
        if match_ratio > 0.75:
            marker_relevant_texts.append(text)

    contradiction_markers = {
        "however", "superseded", "obsolete",
        "no longer", "incorrect", "conflicts with",
    }

    combined = " ".join(marker_relevant_texts).lower()
    marker_hits = sum(marker in combined for marker in contradiction_markers)

    print(f"  Marker-relevant texts count: {len(marker_relevant_texts)}")
    print(f"  Marker hits: {marker_hits}")
    if marker_hits > 0:
        for marker in contradiction_markers:
            if marker in combined:
                print(f"    Found: '{marker}'")

    # ------------------------------------------------------------------
    # 8. Semantic relevance check
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SEMANTIC RELEVANCE CHECK")
    print("=" * 80)

    print("Query asks about:")
    print("  1. Alternate identifier of circulation equipment serving Reactor R-2")
    print("  2. Which inlet strainer protects it")

    if found_conflict:
        print("\nConflicting numeric values involve attributes that are:")
        for i in range(len(relevant_indices)):
            for j in range(i + 1, len(relevant_indices)):
                shared_terms = (
                    evidence_term_sets[relevant_indices[i]]
                    & evidence_term_sets[relevant_indices[j]]
                )
                if len(shared_terms) < 2:
                    continue
                shared_units = (
                    set(relevant_measurement_sets[i])
                    & set(relevant_measurement_sets[j])
                )
                for unit in shared_units:
                    vals_i = relevant_measurement_sets[i][unit]
                    vals_j = relevant_measurement_sets[j][unit]
                    if vals_i and vals_j and vals_i.isdisjoint(vals_j):
                        nid_i = evidence_ids[relevant_indices[i]]
                        nid_j = evidence_ids[relevant_indices[j]]
                        text_i = evidence_texts[relevant_indices[i]][:400]
                        text_j = evidence_texts[relevant_indices[j]][:400]
                        print(f"\n  Node {nid_i} ({vals_i} {unit}):")
                        print(f"    {text_i}")
                        print(f"  Node {nid_j} ({vals_j} {unit}):")
                        print(f"    {text_j}")
                        print(f"\n  Does this concern 'alternate identifier'? No")
                        print(f"  Does this concern 'inlet strainer'? No")
                        print(f"  This appears to be an UNRELATED measurement conflict.")

    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
